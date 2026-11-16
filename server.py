import asyncio
import datetime
import json
import os
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict

import jwt
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.stats import kurtosis, skew

import database
from database import Employee, get_db, init_db, hash_password, verify_password, active_db_type

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# =====================================================================
# 1. SENSOR INGESTION LAYER (Driver / Adapter Pattern)
# =====================================================================

class SensorSource(ABC):
    """Abstract interface guaranteeing seamless migration to physical hardware sensors."""
    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class FileStreamSensorSource(SensorSource):
    """Trace-driven software streaming mock emulating an ADC sampling buffer."""
    def __init__(self, file_path: str, chunk_size: int = 512, sample_delay: float = 1.0):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.sample_delay = sample_delay
        self.cursor = 0
        self.data: np.ndarray = np.array([], dtype=np.float32)
        self.total_samples = 0
        self._ensure_and_load_data()

    def _ensure_and_load_data(self) -> None:
        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) < 10:
            print(f"[Sensor] Trace '{self.file_path}' not found or empty. Generating calibrated benchmark...")
            np.random.seed(42)
            normal_1 = np.random.normal(loc=0.0, scale=0.18, size=5000)
            t = np.linspace(0, 1, 3000)
            mild = np.random.normal(0, 0.22, 3000) + 0.35 * np.sin(2 * np.pi * 120 * t)
            anomaly = np.random.normal(loc=0.0, scale=0.45, size=4000)
            shock_idx = np.random.choice(4000, 80, replace=False)
            anomaly[shock_idx] += np.random.uniform(2.0, 3.5, 80) * np.random.choice([-1, 1], 80)
            normal_2 = np.random.normal(loc=0.0, scale=0.18, size=5000)
            combined = np.concatenate([normal_1, mild, anomaly, normal_2]).astype(np.float32)
            pd.DataFrame({"vibration_amplitude": combined}).to_csv(self.file_path, index=False)

        try:
            df = pd.read_csv(self.file_path, sep=r"[\s,;]+", engine="python", on_bad_lines="skip")
            series = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
            self.data = series.to_numpy(dtype=np.float32)
        except Exception as e:
            print(f"[Sensor] Error reading {self.file_path}: {e}. Fallback to numpy...")
            try:
                self.data = np.loadtxt(self.file_path, delimiter=",", dtype=np.float32)
            except Exception:
                self.data = np.zeros(self.chunk_size * 10, dtype=np.float32)

        self.total_samples = len(self.data)
        self.cursor = 0
        print(f"[Sensor] Loaded {self.total_samples} samples from '{self.file_path}'.")

    def read_frame(self) -> Optional[np.ndarray]:
        if self.total_samples == 0:
            return None

        if self.cursor + self.chunk_size > self.total_samples:
            self.cursor = 0  # Circular stream replay

        frame = self.data[self.cursor : self.cursor + self.chunk_size]
        self.cursor += self.chunk_size
        return frame

    def close(self) -> None:
        self.data = np.array([], dtype=np.float32)


class HardwareSensorSource(SensorSource):
    """Driver stub for physical edge buses (I2C/SPI accelerometer or I2S mic)."""
    def __init__(self, bus_id: int = 1, device_address: int = 0x68, chunk_size: int = 512):
        self.bus_id = bus_id
        self.device_address = device_address
        self.chunk_size = chunk_size
        print(f"[HardwareDriver] Initialized I2C Bus {self.bus_id} at {hex(self.device_address)}")

    def read_frame(self) -> Optional[np.ndarray]:
        return np.random.normal(0, 0.05, self.chunk_size).astype(np.float32)

    def close(self) -> None:
        print("[HardwareDriver] I2C Bus released.")


# =====================================================================
# 2. APPROXIMATE MAC HARDWARE CORE (Bit-Accurate Emulation)
# =====================================================================

class ApproximateMACEngine:
    """
    Simulates fixed-point 16-bit MAC units with dynamically switchable bit-truncation.
    - Exact: 1.00x energy (16 active bits)
    - Mild Approx: 0.48x energy (10 active bits, lower 6 bits masked)
    - Aggressive Approx: 0.22x energy (6 active bits, lower 10 bits masked)
    """
    def __init__(self):
        self.mode = "aggressive_approx"
        self.ENERGY_COEFFICIENTS = {
            "exact": 1.00,
            "mild_approx": 0.48,
            "aggressive_approx": 0.22
        }
        self.cumulative_energy = 0.0
        self.cumulative_ops = 0

    def set_precision_mode(self, mode: str) -> None:
        if mode in self.ENERGY_COEFFICIENTS:
            self.mode = mode

    def execute_mac_vector(self, vector_a: np.ndarray, vector_b: np.ndarray) -> Tuple[float, float]:
        n_ops = len(vector_a)
        a_int = np.clip(vector_a * 1024, -32768, 32767).astype(np.int32)
        b_int = np.clip(vector_b * 1024, -32768, 32767).astype(np.int32)

        # 1. Exact reference
        exact_accum = int(np.dot(a_int, b_int))
        exact_res = float(exact_accum / (1024.0 * 1024.0))

        # 2. Hardware bit-truncation
        if self.mode == "exact":
            approx_accum = exact_accum
        elif self.mode == "mild_approx":
            mask = ~((1 << 6) - 1)
            approx_accum = int(np.dot(a_int & mask, b_int & mask))
        elif self.mode == "aggressive_approx":
            mask = ~((1 << 10) - 1)
            approx_accum = int(np.dot(a_int & mask, b_int & mask))
        else:
            approx_accum = exact_accum

        approx_res = float(approx_accum / (1024.0 * 1024.0))

        # 3. Energy accounting
        self.cumulative_ops += n_ops
        self.cumulative_energy += n_ops * self.ENERGY_COEFFICIENTS[self.mode]

        return approx_res, exact_res

    def compute_energy_savings_pct(self) -> float:
        if self.cumulative_ops == 0:
            return 0.0
        baseline = self.cumulative_ops * 1.00
        return ((baseline - self.cumulative_energy) / baseline) * 100.0


# =====================================================================
# 3. AI PRECISION CONTROLLER & ANOMALY DETECTOR
# =====================================================================

class AIPrecisionController:
    """Evaluates incoming signal stability using trained XGBoost classifier or calibrated fallback."""
    def __init__(
        self,
        kurtosis_threshold: float = 3.5,
        variance_threshold: float = 1.8,
        model_path: str = "precision_model.json"
    ):
        self.kurtosis_threshold = kurtosis_threshold
        self.variance_threshold = variance_threshold
        self.model_path = model_path
        self.xgb_model = None
        self.mode_names = ["aggressive_approx", "mild_approx", "exact"]
        self.load_model()

    def load_model(self) -> bool:
        if XGB_AVAILABLE and os.path.exists(self.model_path):
            try:
                self.xgb_model = xgb.XGBClassifier()
                self.xgb_model.load_model(self.model_path)
                print(f"[AIController] Successfully loaded pre-trained XGBoost policy from '{self.model_path}'.")
                return True
            except Exception as e:
                print(f"[AIController] Failed to load model '{self.model_path}': {e}. Using heuristic mode.")
                self.xgb_model = None
        return False

    def evaluate_frame(self, frame: np.ndarray) -> Tuple[str, float]:
        rms = float(np.sqrt(np.mean(frame**2)))
        peak = float(np.max(np.abs(frame)))
        kurt_val = float(kurtosis(frame))
        skew_val = float(skew(frame))
        var_val = float(np.var(frame))
        crest_factor = float(peak / (rms + 1e-7))

        if self.xgb_model is not None:
            try:
                x_input = np.array([[rms, peak, kurt_val, skew_val, var_val, crest_factor]], dtype=np.float32)
                pred_idx = int(self.xgb_model.predict(x_input)[0])
                probs = self.xgb_model.predict_proba(x_input)[0]
                conf = round(float(probs[pred_idx]) * 100, 2)
                return self.mode_names[pred_idx], conf
            except Exception:
                pass

        if kurt_val > self.kurtosis_threshold or var_val > self.variance_threshold or crest_factor > 4.5 or rms > 0.40:
            return "exact", 96.0
        elif kurt_val > (self.kurtosis_threshold * 0.70) or var_val > (self.variance_threshold * 0.40) or crest_factor > 2.8:
            return "mild_approx", 92.5
        else:
            return "aggressive_approx", 98.0


class VibrationAnomalyDetector:
    """Executes filter projection and anomaly scoring via the approximate MAC engine."""
    def __init__(self, window_size: int = 512):
        self.window_size = window_size
        t = np.linspace(0, np.pi, window_size, dtype=np.float32)
        self.filter_weights = (np.sin(t) / np.sum(np.sin(t))).astype(np.float32)

    def process_window(self, frame: np.ndarray, engine: ApproximateMACEngine) -> dict:
        approx_val, exact_val = engine.execute_mac_vector(frame, self.filter_weights)
        quant_error = abs(exact_val - approx_val)
        anomaly_score = float(np.max(np.abs(frame)))
        is_anomaly = (engine.mode == "exact") or (anomaly_score > 1.25)

        return {
            "mac_approx": approx_val,
            "mac_exact": exact_val,
            "quant_error": quant_error,
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly
        }


# =====================================================================
# 4. FASTAPI APPLICATION, CONFIG & ANOMALY STORAGE
# =====================================================================

app = FastAPI(title="Adaptive Approximate MAC Accelerator Telemetry Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


class PipelineConfig(BaseModel):
    source_type: str = "mock"
    dataset_file: str = "test_trace.csv"
    chunk_size: int = 512
    sampling_delay: float = 1.0  # Slow, calm, human-observable inspection speed (1 frame per second)
    kurtosis_threshold: float = 3.5
    variance_threshold: float = 1.8


class SpeedConfig(BaseModel):
    sampling_delay: float


class FeatureVector(BaseModel):
    rms: float
    peak: float
    kurtosis: float
    skewness: float
    variance: float
    crest_factor: float


config = PipelineConfig()
stream_active = True
global_controller = AIPrecisionController(
    kurtosis_threshold=config.kurtosis_threshold,
    variance_threshold=config.variance_threshold
)

# Persistent Anomaly Events Storage for Analytics Page
ANOMALY_LOG_FILE = "anomaly_events.json"
anomaly_events: List[dict] = []


def load_anomaly_events():
    global anomaly_events
    if os.path.exists(ANOMALY_LOG_FILE):
        try:
            with open(ANOMALY_LOG_FILE, "r", encoding="utf-8") as f:
                anomaly_events = json.load(f)
        except Exception:
            anomaly_events = []


def save_anomaly_events():
    try:
        with open(ANOMALY_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(anomaly_events[:500], f, indent=2)
    except Exception:
        pass


load_anomaly_events()


def sync_static_assets():
    """Ensure assets are in static/ so both direct and static mounts serve correctly."""
    os.makedirs("static", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs(os.path.join("static", "uploads"), exist_ok=True)
    
    # Public assets only - protected application pages must NOT be directly reachable without auth
    public_files = [
        "index.html",
        "login.html",
        "register.html",
        "acid-squares.css",
        "acid-squares.js",
        "auth_guard.js",
        "cinematic_transitions.js",
        "product_interface.svg",
        "sounds_bundle.js",
        "marimba.wav",
        "glass.wav",
        "bell.wav"
    ]
    for pf in public_files:
        if os.path.exists(pf):
            target = os.path.join("static", pf)
            shutil.copyfile(pf, target)

    # Remove any sensitive protected pages from static/ so direct bypass is impossible
    protected = [
        "monitor.html", "profile.html", "spectral.html", "bit_inspector.html",
        "ai_policy.html", "energy.html", "config.html", "analytics.html"
    ]
    for pp in protected:
        static_p = os.path.join("static", pp)
        if os.path.exists(static_p):
            try:
                os.remove(static_p)
            except Exception:
                pass

    # Sync default avatar if present
    if os.path.exists("uploads/default_avatar.svg"):
        shutil.copyfile("uploads/default_avatar.svg", "static/uploads/default_avatar.svg")
    elif os.path.exists("static/uploads/default_avatar.svg"):
        shutil.copyfile("static/uploads/default_avatar.svg", "uploads/default_avatar.svg")


sync_static_assets()
database.init_db()


def ensure_initial_dataset():
    """Validates existence of industrial dataset and model, generating/training if missing."""
    dataset_file = "industrial_vibration_dataset.csv"
    model_file = "precision_model.json"

    if (not os.path.exists(dataset_file) or os.path.getsize(dataset_file) < 100000) and os.path.exists("industrial_vibration_master.csv"):
        shutil.copyfile("industrial_vibration_master.csv", dataset_file)

    if not os.path.exists(dataset_file) or os.path.getsize(dataset_file) < 100:
        print("[Startup] Cleaned dataset missing or empty. Running dataset cleaner...")
        try:
            from clean_dataset import clean_entire_dataset
            clean_entire_dataset(dataset_file)
        except Exception as e:
            print(f"[Startup] Error building dataset: {e}")

    if not os.path.exists(model_file) or os.path.getsize(model_file) < 100:
        print("[Startup] Trained model missing. Triggering XGBoost trainer...")
        try:
            from train_model import train_precision_model
            train_precision_model(dataset_file, model_file)
            global_controller.load_model()
        except Exception as e:
            print(f"[Startup] Error training model: {e}")


ensure_initial_dataset()


# =====================================================================
# 5. REST API & STATIC SERVING ENDPOINTS
# =====================================================================

# Protected application pages - ONLY accessible after employee login
PROTECTED_PAGES = {
    "monitor",
    "profile",
    "spectral",
    "bit_inspector",
    "ai_policy",
    "energy",
    "analytics",
    "config"
}


@app.get("/")
async def get_index():
    if os.path.exists("login.html"):
        return FileResponse("login.html")
    elif os.path.exists("index.html"):
        return FileResponse("index.html")
    return FileResponse("static/index.html")


@app.get("/login")
async def get_login_alias():
    if os.path.exists("login.html"):
        return FileResponse("login.html")
    return RedirectResponse(url="/", status_code=303)


@app.get("/register")
@app.get("/create-account")
async def get_register_alias():
    if os.path.exists("register.html"):
        return FileResponse("register.html")
    return RedirectResponse(url="/", status_code=303)


@app.get("/privacy")
async def get_privacy_alias():
    if os.path.exists("privacy.html"):
        return FileResponse("privacy.html")
    return RedirectResponse(url="/privacy.html", status_code=303)


@app.get("/terms")
@app.get("/terms-and-conditions")
async def get_terms_alias():
    if os.path.exists("terms.html"):
        return FileResponse("terms.html")
    return RedirectResponse(url="/terms.html", status_code=303)



@app.get("/{page}.html")
async def get_html_page(page: str, request: Request):
    """
    Enforces that application pages cannot be opened without being logged in.
    Redirects unauthenticated visitors directly to the Employee Login page.
    """
    if page in PROTECTED_PAGES:
        db = next(get_db())
        try:
            emp = get_current_employee_from_request(request, db)
            if not emp:
                return RedirectResponse(url="/login.html?error=login_required", status_code=303)
        finally:
            db.close()

    root_path = f"{page}.html"
    static_path = os.path.join("static", f"{page}.html")
    if os.path.exists(root_path):
        return FileResponse(root_path)
    elif os.path.exists(static_path):
        return FileResponse(static_path)
    return RedirectResponse(url="/login.html", status_code=303)


@app.get("/static/{page}.html")
async def get_static_html_page(page: str, request: Request):
    """Intercept direct /static/*.html requests to enforce login restriction."""
    if page in PROTECTED_PAGES:
        db = next(get_db())
        try:
            emp = get_current_employee_from_request(request, db)
            if not emp:
                return RedirectResponse(url="/login.html?error=login_required", status_code=303)
        finally:
            db.close()
    if os.path.exists(f"{page}.html"):
        return FileResponse(f"{page}.html")
    return RedirectResponse(url="/login.html", status_code=303)


@app.get("/auth_guard.js")
async def get_auth_guard_js():
    if os.path.exists("auth_guard.js"):
        return FileResponse("auth_guard.js", media_type="application/javascript")
    return FileResponse("static/auth_guard.js", media_type="application/javascript")


@app.get("/cinematic_transitions.js")
async def get_cinematic_transitions_js():
    if os.path.exists("cinematic_transitions.js"):
        return FileResponse("cinematic_transitions.js", media_type="application/javascript")
    return FileResponse("static/cinematic_transitions.js", media_type="application/javascript")


@app.get("/acid-squares.css")
async def get_acid_squares_css():
    if os.path.exists("acid-squares.css"):
        return FileResponse("acid-squares.css", media_type="text/css")
    return FileResponse("static/acid-squares.css", media_type="text/css")


@app.get("/acid-squares.js")
async def get_acid_squares_js():
    if os.path.exists("acid-squares.js"):
        return FileResponse("acid-squares.js", media_type="application/javascript")
    return FileResponse("static/acid-squares.js", media_type="application/javascript")


@app.get("/product_interface.svg")
async def get_product_interface_svg():
    if os.path.exists("product_interface.svg"):
        return FileResponse("product_interface.svg", media_type="image/svg+xml")
    return FileResponse("static/product_interface.svg", media_type="image/svg+xml")


@app.get("/sounds_bundle.js")
async def get_sounds_bundle_js():
    if os.path.exists("sounds_bundle.js"):
        return FileResponse("sounds_bundle.js", media_type="application/javascript")
    return FileResponse("static/sounds_bundle.js", media_type="application/javascript")


@app.get("/{sound}.wav")
async def get_wav_audio(sound: str):
    root_path = f"{sound}.wav"
    static_path = os.path.join("static", f"{sound}.wav")
    if os.path.exists(root_path):
        return FileResponse(root_path, media_type="audio/wav")
    elif os.path.exists(static_path):
        return FileResponse(static_path, media_type="audio/wav")
    return {"error": "Audio file not found"}


@app.get("/api/config")
async def get_config():
    return config


@app.post("/api/config")
async def update_config(new_config: PipelineConfig):
    global config
    config = new_config
    global_controller.kurtosis_threshold = config.kurtosis_threshold
    global_controller.variance_threshold = config.variance_threshold
    return {"status": "success", "config": config}


@app.post("/api/speed")
async def update_speed(speed: SpeedConfig):
    global config
    config.sampling_delay = max(0.05, min(5.0, speed.sampling_delay))
    print(f"[Engine] Set telemetry delay to {config.sampling_delay}s")
    return {"status": "success", "sampling_delay": config.sampling_delay}


@app.post("/api/toggle-stream")
async def toggle_stream():
    global stream_active
    stream_active = not stream_active
    return {"streaming_active": stream_active}


# =====================================================================
# ANALYTICS ENDPOINTS (Fault & Anomaly Tracking)
# =====================================================================

@app.get("/api/analytics/events")
async def get_analytics_events():
    """Provides historical and live anomaly spikes for the Analytics table."""
    critical_cnt = sum(1 for e in anomaly_events if e.get("severity") == "CRITICAL")
    warning_cnt = sum(1 for e in anomaly_events if e.get("severity") == "WARNING")
    return {
        "status": "success",
        "total_count": len(anomaly_events),
        "critical_count": critical_cnt,
        "warning_count": warning_cnt,
        "events": anomaly_events
    }


@app.post("/api/analytics/clear")
async def clear_analytics_events():
    """Clears stored anomaly table events."""
    global anomaly_events
    anomaly_events = []
    save_anomaly_events()
    return {"status": "success", "message": "Anomaly event log cleared."}


@app.get("/api/analytics/export")
async def export_analytics_csv():
    """Exports all recorded anomaly events as CSV."""
    if not anomaly_events:
        return PlainTextResponse("No anomalies recorded yet.\n", media_type="text/csv")
    df_events = pd.DataFrame(anomaly_events)
    csv_str = df_events.to_csv(index=False)
    return PlainTextResponse(
        csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=fault_anomalies_log.csv"}
    )


@app.get("/api/dataset/samples")
async def get_dataset_samples():
    filename = "industrial_vibration_dataset.csv"
    if not os.path.exists(filename) or os.path.getsize(filename) < 100:
        ensure_initial_dataset()

    try:
        df = pd.read_csv(filename)
        def get_category_row(mode_name: str, fallback_dict: dict):
            subset = df[df["target_mode"] == mode_name]
            if not subset.empty:
                return subset.iloc[0].to_dict()
            return fallback_dict

        sample_normal = get_category_row(
            "aggressive_approx",
            {"rms": 0.082, "peak": 0.285, "kurtosis": 1.45, "skewness": 0.02, "variance": 0.007, "crest_factor": 3.47, "target_mode": "aggressive_approx"}
        )
        sample_mild = get_category_row(
            "mild_approx",
            {"rms": 0.245, "peak": 0.780, "kurtosis": 2.65, "skewness": 0.35, "variance": 0.060, "crest_factor": 3.18, "target_mode": "mild_approx"}
        )
        sample_severe = get_category_row(
            "exact",
            {"rms": 0.650, "peak": 3.450, "kurtosis": 8.20, "skewness": 1.15, "variance": 0.422, "crest_factor": 5.31, "target_mode": "exact"}
        )

        return {
            "status": "success",
            "samples": [
                {"category": "Normal Operation (Healthy Bearing)", **sample_normal},
                {"category": "Mild Degradation (Inner Race Wear)", **sample_mild},
                {"category": "Severe Anomaly (Impeller Cavitation / Outer Crack)", **sample_severe}
            ]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/model/predict")
async def predict_sample(features: FeatureVector):
    x_input = np.array([[features.rms, features.peak, features.kurtosis,
                         features.skewness, features.variance, features.crest_factor]], dtype=np.float32)

    if global_controller.xgb_model is not None:
        try:
            pred_idx = int(global_controller.xgb_model.predict(x_input)[0])
            probs = global_controller.xgb_model.predict_proba(x_input)[0].tolist()
            predicted_mode = global_controller.mode_names[pred_idx]
            confidence = round(float(probs[pred_idx]) * 100, 2)
        except Exception:
            predicted_mode, confidence = global_controller.evaluate_frame(
                np.array([features.rms, features.peak, features.variance], dtype=np.float32)
            )
    else:
        if features.kurtosis > config.kurtosis_threshold or features.variance > config.variance_threshold:
            predicted_mode = "exact"
            confidence = 95.0
        elif features.kurtosis > (config.kurtosis_threshold * 0.70):
            predicted_mode = "mild_approx"
            confidence = 92.0
        else:
            predicted_mode = "aggressive_approx"
            confidence = 98.0

    energy_savings = 78.0 if predicted_mode == "aggressive_approx" else (52.0 if predicted_mode == "mild_approx" else 0.0)

    return {
        "predicted_mode": predicted_mode,
        "confidence": confidence,
        "energy_savings_pct": energy_savings
    }


# =====================================================================
# 6. WEBSOCKET REAL-TIME TELEMETRY PIPELINE
# =====================================================================

@app.websocket("/ws/telemetry")
async def telemetry_stream(websocket: WebSocket):
    await websocket.accept()

    current_source_file = config.dataset_file
    current_source_type = config.source_type

    def create_source(source_type: str, file_path: str) -> SensorSource:
        if source_type == "hardware":
            return HardwareSensorSource(chunk_size=config.chunk_size)
        else:
            return FileStreamSensorSource(
                file_path=file_path,
                chunk_size=config.chunk_size,
                sample_delay=config.sampling_delay
            )

    source = create_source(current_source_type, current_source_file)
    mac_engine = ApproximateMACEngine()
    detector = VibrationAnomalyDetector(window_size=config.chunk_size)
    frame_id = 0

    try:
        while True:
            if config.dataset_file != current_source_file or config.source_type != current_source_type:
                source.close()
                current_source_file = config.dataset_file
                current_source_type = config.source_type
                source = create_source(current_source_type, current_source_file)

            if not stream_active:
                await asyncio.sleep(0.1)
                continue

            frame = source.read_frame()
            if frame is None or len(frame) == 0:
                await asyncio.sleep(0.05)
                continue

            frame_id += 1

            # 1. AI Controller decides arithmetic precision mode
            selected_mode, confidence = global_controller.evaluate_frame(frame)
            mac_engine.set_precision_mode(selected_mode)

            # 2. Approximate MAC executes feature projection
            results = detector.process_window(frame, mac_engine)

            # 3. Downsample frame for smooth 50-point frontend display
            step = max(1, len(frame) // 50)
            downsampled = [round(float(v), 3) for v in frame[::step]]

            # 4. Check for fault anomaly spike to store in analytics table
            new_anomaly = None
            if results["is_anomaly"] or selected_mode == "exact":
                kurt_val = float(kurtosis(frame))
                peak_val = float(np.max(np.abs(frame)))
                rms_val = float(np.sqrt(np.mean(frame**2)))
                cf_val = float(peak_val / (rms_val + 1e-7))

                new_anomaly = {
                    "id": len(anomaly_events) + 1,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "frame_id": frame_id,
                    "mode": selected_mode,
                    "confidence": confidence,
                    "kurtosis": round(kurt_val, 2),
                    "peak": round(peak_val, 3),
                    "rms": round(rms_val, 3),
                    "crest_factor": round(cf_val, 2),
                    "variance": round(float(np.var(frame)), 3),
                    "quant_error": round(results["quant_error"], 4),
                    "scenario": os.path.basename(config.dataset_file),
                    "severity": "CRITICAL" if kurt_val > 4.5 or peak_val > 2.0 else "WARNING",
                    "action": "Escalated to 16-Bit Exact MAC"
                }

                # Avoid duplicate identical frame spamming
                if not anomaly_events or (frame_id - anomaly_events[0]["frame_id"] >= 1):
                    anomaly_events.insert(0, new_anomaly)
                    if len(anomaly_events) > 500:
                        anomaly_events.pop()
                    save_anomaly_events()

            # 5. Construct telemetry message
            payload = {
                "frame_id": frame_id,
                "mode": selected_mode,
                "confidence": confidence,
                "kurtosis": round(float(kurtosis(frame)), 2),
                "variance": round(float(np.var(frame)), 3),
                "anomaly_detected": results["is_anomaly"],
                "mac_output": round(results["mac_approx"], 4),
                "exact_reference": round(results["mac_exact"], 4),
                "quant_error": round(results["quant_error"], 4),
                "energy_saved_pct": round(mac_engine.compute_energy_savings_pct(), 1),
                "total_macs": mac_engine.cumulative_ops,
                "signal_samples": downsampled,
                "new_anomaly": new_anomaly
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(config.sampling_delay)

    except WebSocketDisconnect:
        pass
    finally:
        source.close()



# =====================================================================
# 7. EMPLOYEE AUTHENTICATION & PROFILE SYSTEM (PostgreSQL + Bcrypt)
# =====================================================================

JWT_SECRET = "approxmac-super-secure-jwt-secret-key-32bytes-min-2026"
JWT_ALGORITHM = "HS256"


class EmployeeRegisterRequest(BaseModel):
    name: str
    employee_id: str
    phone: str
    email: str
    password: str


class EmployeeLoginRequest(BaseModel):
    employee_id: str
    password: str


def create_jwt_token(employee: Employee) -> str:
    payload = {
        "sub": employee.employee_id,
        "name": employee.name,
        "email": employee.email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_employee_from_request(request: Request, db):
    auth_header = request.headers.get("Authorization")
    token = None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
    elif "session_token" in request.cookies:
        token = request.cookies.get("session_token")

    if not token:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        emp_id = payload.get("sub")
        if not emp_id:
            return None
        return db.query(Employee).filter(Employee.employee_id == emp_id).first()
    except Exception:
        return None


@app.get("/api/auth/db-status")
async def get_db_status():
    """Returns whether running on PostgreSQL or SQLite fallback."""
    return {
        "status": "success",
        "database": database.active_db_type
    }


@app.post("/api/auth/register")
async def register_employee(req: EmployeeRegisterRequest):
    """
    Employee account creation.
    Validates fields, verifies uniqueness, and encrypts password using bcrypt.
    """
    db = next(get_db())
    try:
        name = req.name.strip()
        emp_id = req.employee_id.strip()
        phone = req.phone.strip()
        email = req.email.strip().lower()
        password = req.password

        if not name or not emp_id or not phone or not email or not password:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "All fields are required. Please fill out all fields."}
            )

        if db.query(Employee).filter(Employee.employee_id == emp_id).first():
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Employee ID '{emp_id}' is already registered."}
            )

        if db.query(Employee).filter(Employee.email == email).first():
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Employee email '{email}' is already registered."}
            )

        hashed_pw = hash_password(password)

        new_emp = Employee(
            name=name,
            employee_id=emp_id,
            phone=phone,
            email=email,
            password_hash=hashed_pw,
            profile_pic="/static/uploads/default_avatar.svg"
        )
        db.add(new_emp)
        db.commit()
        db.refresh(new_emp)

        return {
            "status": "success",
            "message": "Employee account created successfully! Please log in.",
            "employee_id": new_emp.employee_id,
            "name": new_emp.name
        }
    except Exception as e:
        db.rollback()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Registration failed: {str(e)}"}
        )
    finally:
        db.close()


@app.post("/api/auth/login")
async def login_employee(req: EmployeeLoginRequest, response: Response):
    """
    Employee login.
    Verifies credentials using bcrypt.checkpw, issues session cookie and JWT.
    """
    db = next(get_db())
    try:
        ident = req.employee_id.strip()
        password = req.password

        emp = db.query(Employee).filter(
            (Employee.employee_id == ident) | (Employee.email == ident.lower())
        ).first()

        if not emp:
            return JSONResponse(
                status_code=401,
                content={"status": "error", "message": "Employee not found. Please verify your Employee ID or Email."}
            )

        if not verify_password(password, emp.password_hash):
            return JSONResponse(
                status_code=401,
                content={"status": "error", "message": "Incorrect password. Please try again."}
            )

        token = create_jwt_token(emp)
        response.set_cookie(
            key="session_token",
            value=token,
            httponly=False,
            max_age=7 * 24 * 3600,
            samesite="lax",
            path="/"
        )

        return {
            "status": "success",
            "message": f"Welcome back, {emp.name}!",
            "token": token,
            "employee": emp.to_dict()
        }
    finally:
        db.close()


@app.get("/api/auth/me")
async def get_my_profile(request: Request):
    """Retrieves current logged-in employee profile."""
    db = next(get_db())
    try:
        emp = get_current_employee_from_request(request, db)
        if not emp:
            return JSONResponse(
                status_code=401,
                content={"status": "unauthorized", "message": "No active employee session. Please log in."}
            )

        return {
            "status": "success",
            "employee": emp.to_dict(),
            "database": database.active_db_type
        }
    finally:
        db.close()


@app.post("/api/auth/upload-avatar")
async def upload_avatar(
    request: Request,
    file: UploadFile = File(...),
    employee_id: Optional[str] = Form(None)
):
    """
    Handles employee profile picture upload.
    Saves image to uploads/ and static/uploads/, updates Employee.profile_pic in PostgreSQL.
    """
    db = next(get_db())
    try:
        emp = get_current_employee_from_request(request, db)
        if not emp:
            return JSONResponse(
                status_code=401,
                content={"status": "unauthorized", "message": "Authentication required. Please log in to upload avatar."}
            )

        ext = os.path.splitext(file.filename)[1].lower() if file.filename else ".jpg"
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg"]:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Supported image formats: PNG, JPG, JPEG, WEBP, GIF, SVG."}
            )

        filename = f"avatar_{emp.employee_id}_{int(time.time())}{ext}"
        upload_dir = "uploads"
        static_upload_dir = os.path.join("static", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(static_upload_dir, exist_ok=True)

        target_file = os.path.join(upload_dir, filename)
        static_target_file = os.path.join(static_upload_dir, filename)

        contents = await file.read()
        with open(target_file, "wb") as f:
            f.write(contents)
        with open(static_target_file, "wb") as f:
            f.write(contents)

        avatar_url = f"/static/uploads/{filename}"
        emp.profile_pic = avatar_url
        db.commit()
        db.refresh(emp)

        return {
            "status": "success",
            "message": "Profile picture updated successfully!",
            "profile_pic": avatar_url,
            "employee": emp.to_dict()
        }
    except Exception as e:
        db.rollback()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Upload failed: {str(e)}"}
        )
    finally:
        db.close()


@app.post("/api/auth/logout")
async def logout_employee(response: Response):
    """Logs out employee and clears session cookie."""
    response.delete_cookie(key="session_token", path="/")
    return {"status": "success", "message": "Logged out successfully."}


os.makedirs("uploads", exist_ok=True)
os.makedirs(os.path.join("static", "uploads"), exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    sync_static_assets()
    port = int(os.environ.get("PORT", 8000))
    print("\n" + "=" * 60)
    print(f"Dashboard URL: http://0.0.0.0:{port} or http://127.0.0.1:{port}")
    print("=" * 60 + "\n")
    uvicorn.run("server:app", host="0.0.0.0", port=port)
