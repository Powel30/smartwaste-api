"""
SmartWaste IoT Cloud API
Kyambogo University — Capstone Project
Author : Ssekamatte Powel Kelly (23/U/BIE/2640/PE)
Supervisor: Mr. Okello Wayne

Deploy on: Render / Railway / any VPS (Python 3.10+)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import joblib, os, json, logging
import numpy as np

# ── Optional: real DB swap ──────────────────────────────────────────────────
# Replace in-memory store with SQLite / PostgreSQL when going to production
# from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
CORS(app)  # Allow React dashboard on any origin
logging.basicConfig(level=logging.INFO)

# ── In-memory data store (replace with DB in production) ───────────────────
bins_data = {
    "BIN-01": {"ward": "ICU",        "type": "Infectious",  "capacity_cm": 100,
               "fill_pct": 0, "temp": 24.0, "gas_ppm": 150, "weight_kg": 0,
               "last_seen": None, "history": []},
    "BIN-02": {"ward": "Surgery",    "type": "Sharps",      "capacity_cm": 60,
               "fill_pct": 0, "temp": 23.5, "gas_ppm": 120, "weight_kg": 0,
               "last_seen": None, "history": []},
    "BIN-03": {"ward": "Pharmacy",   "type": "Chemical",    "capacity_cm": 80,
               "fill_pct": 0, "temp": 25.0, "gas_ppm": 200, "weight_kg": 0,
               "last_seen": None, "history": []},
    "BIN-04": {"ward": "General",    "type": "General",     "capacity_cm": 120,
               "fill_pct": 0, "temp": 22.0, "gas_ppm": 100, "weight_kg": 0,
               "last_seen": None, "history": []},
    "BIN-05": {"ward": "Radiology",  "type": "Radioactive", "capacity_cm": 40,
               "fill_pct": 0, "temp": 23.0, "gas_ppm": 90,  "weight_kg": 0,
               "last_seen": None, "history": []},
    "BIN-06": {"ward": "Paediatric", "type": "Infectious",  "capacity_cm": 80,
               "fill_pct": 0, "temp": 24.5, "gas_ppm": 130, "weight_kg": 0,
               "last_seen": None, "history": []},
}

# ── ML model (loaded once at startup) ──────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "waste_model.pkl")
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    app.logger.info(f"ML model loaded from {MODEL_PATH}")
else:
    app.logger.warning("No ML model found — using heuristic predictions")


def predict_hours_to_full(fill_pct: float, gas_ppm: float,
                           temp: float, weight_kg: float) -> float:
    """Returns estimated hours until bin is full."""
    if model:
        features = np.array([[fill_pct, gas_ppm, temp, weight_kg]])
        return float(model.predict(features)[0])
    # Heuristic fallback: linear fill rate ~3.2 %/hr
    rate = max(0.5, (gas_ppm / 500) * 4.0)
    return round(max(0, (100 - fill_pct) / rate), 1)


def risk_label(hours: float) -> str:
    if hours < 4:   return "CRITICAL"
    if hours < 12:  return "HIGH"
    if hours < 24:  return "MEDIUM"
    return "LOW"


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════

# ── POST /api/ingest  ← called by ESP32 gateway node ───────────────────────
@app.route("/api/ingest", methods=["POST"])
def ingest():
    """
    Payload from gateway (JSON):
    {
      "bin_id"     : "BIN-01",
      "distance_cm": 32.5,
      "temp_c"     : 24.3,
      "humidity"   : 65.1,
      "gas_ppm"    : 280,
      "weight_kg"  : 3.2
    }
    """
    data = request.get_json(force=True)
    bin_id = data.get("bin_id")

    if bin_id not in bins_data:
        return jsonify({"error": "Unknown bin_id"}), 400

    b = bins_data[bin_id]
    cap = b["capacity_cm"]
    dist = float(data.get("distance_cm", cap))
    fill_pct = round(max(0, min(100, (1 - dist / cap) * 100)), 1)

    b["fill_pct"]  = fill_pct
    b["temp"]      = float(data.get("temp_c", b["temp"]))
    b["gas_ppm"]   = float(data.get("gas_ppm", b["gas_ppm"]))
    b["weight_kg"] = float(data.get("weight_kg", b["weight_kg"]))
    b["last_seen"] = datetime.utcnow().isoformat()

    # Save rolling 24-h history (288 = every 5 min for 24 h)
    b["history"].append({
        "ts":       b["last_seen"],
        "fill_pct": fill_pct,
        "temp":     b["temp"],
        "gas_ppm":  b["gas_ppm"],
    })
    b["history"] = b["history"][-288:]

    hours = predict_hours_to_full(fill_pct, b["gas_ppm"], b["temp"], b["weight_kg"])
    app.logger.info(f"[{bin_id}] fill={fill_pct}% gas={b['gas_ppm']} hrs_left={hours}")

    return jsonify({
        "status":        "ok",
        "bin_id":        bin_id,
        "fill_pct":      fill_pct,
        "hours_to_full": hours,
        "risk":          risk_label(hours),
    }), 200


# ── GET /api/bins  ← called by React dashboard ─────────────────────────────
@app.route("/api/bins", methods=["GET"])
def get_bins():
    out = []
    for bin_id, b in bins_data.items():
        hours = predict_hours_to_full(b["fill_pct"], b["gas_ppm"], b["temp"], b["weight_kg"])
        out.append({
            "bin_id":        bin_id,
            "ward":          b["ward"],
            "type":          b["type"],
            "fill_pct":      b["fill_pct"],
            "temp":          b["temp"],
            "gas_ppm":       b["gas_ppm"],
            "weight_kg":     b["weight_kg"],
            "hours_to_full": hours,
            "risk":          risk_label(hours),
            "last_seen":     b["last_seen"],
        })
    return jsonify(out), 200


# ── GET /api/bins/<bin_id>/history  ← chart data ───────────────────────────
@app.route("/api/bins/<bin_id>/history", methods=["GET"])
def get_history(bin_id):
    if bin_id not in bins_data:
        return jsonify({"error": "Unknown bin_id"}), 404
    return jsonify(bins_data[bin_id]["history"]), 200


# ── GET /api/summary  ← KPI cards ──────────────────────────────────────────
@app.route("/api/summary", methods=["GET"])
def summary():
    fills   = [b["fill_pct"]  for b in bins_data.values()]
    gases   = [b["gas_ppm"]   for b in bins_data.values()]
    temps   = [b["temp"]      for b in bins_data.values()]
    critical = sum(1 for f in fills if f >= 80)
    return jsonify({
        "total_bins":     len(bins_data),
        "avg_fill_pct":   round(sum(fills) / len(fills), 1),
        "critical_bins":  critical,
        "max_gas_ppm":    max(gases),
        "avg_temp_c":     round(sum(temps) / len(temps), 1),
        "timestamp":      datetime.utcnow().isoformat(),
    }), 200


# ── GET /api/alerts  ← alert list ──────────────────────────────────────────
@app.route("/api/alerts", methods=["GET"])
def alerts():
    result = []
    for bin_id, b in bins_data.items():
        hours = predict_hours_to_full(b["fill_pct"], b["gas_ppm"], b["temp"], b["weight_kg"])
        risk  = risk_label(hours)
        if risk in ("CRITICAL", "HIGH"):
            result.append({
                "bin_id":        bin_id,
                "ward":          b["ward"],
                "fill_pct":      b["fill_pct"],
                "hours_to_full": hours,
                "risk":          risk,
            })
    return jsonify(result), 200


# ── Health check ────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
