import os
import time

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from threading import FireFighterManager

app = Flask(__name__)
CORS(app)

MODEL_PATH = "../datasets/results/model_weights.pth"
NUM_FIREFIGHTERS = 5
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

manager = FireFighterManager(MODEL_PATH, NUM_FIREFIGHTERS)

# In-memory state for each firefighter
state = {
    i: {"lat": 0.0, "lon": 0.0, "alt": 0.0, "label": None, "live": False, "image_ts": 0}
    for i in range(NUM_FIREFIGHTERS)
}


# ── Ingest endpoints (called by firefighter devices) ────────────────────────

@app.route("/api/image", methods=["POST"])
def receive_image():
    """Receive an image from a firefighter device.
    Expects multipart form: firefighter_id (int) + image (file).
    """
    firefighter_id = int(request.form.get("firefighter_id", 0))
    file = request.files["image"]

    path = os.path.join(UPLOAD_DIR, f"firefighter_{firefighter_id}.jpg")
    file.save(path)

    manager.send_image(path, worker_id=firefighter_id)
    state[firefighter_id]["live"] = True
    state[firefighter_id]["image_ts"] = time.time()

    return jsonify({"status": "ok", "firefighter_id": firefighter_id})


@app.route("/api/gps", methods=["POST"])
def receive_gps():
    """Receive GPS data from a firefighter device.
    Expects JSON: { "firefighter_id": 0, "lat": 33.64, "lon": -117.84, "alt": 0.0 }
    """
    data = request.get_json(force=True)
    fid = int(data.get("firefighter_id", 0))

    state[fid]["lat"] = float(data.get("lat", 0.0))
    state[fid]["lon"] = float(data.get("lon", 0.0))
    state[fid]["alt"] = float(data.get("alt", 0.0))
    state[fid]["live"] = True

    return jsonify({"status": "ok", "firefighter_id": fid})


# ── Frontend endpoints ──────────────────────────────────────────────────────

@app.route("/api/state", methods=["GET"])
def get_state():
    """Polled by the React frontend. Returns the latest data for every firefighter."""
    # Drain any finished inference results into state
    for i in range(NUM_FIREFIGHTERS):
        result = manager.get_result(worker_id=i)
        if result is not None:
            # result is (image_path, numpy_array)
            state[i]["label"] = result[1].tolist()

    firefighters = []
    for i in range(NUM_FIREFIGHTERS):
        s = state[i]
        firefighters.append({
            "id": i,
            "live": s["live"],
            "image_url": f"/api/images/{i}?t={s['image_ts']}" if s["live"] else None,
            "lat": s["lat"],
            "lon": s["lon"],
            "alt": s["alt"],
            "label": s["label"],
        })

    return jsonify({"firefighters": firefighters})


@app.route("/api/images/<int:fid>")
def serve_image(fid):
    """Serve the latest saved image for a firefighter."""
    return send_from_directory(UPLOAD_DIR, f"firefighter_{fid}.jpg")


if __name__ == "__main__":
    app.run(debug=False)
