import os
import time
import threading

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from workers import FireFighterManager

app = Flask(__name__)
CORS(app)

MODEL_PATH = "../datasets/results/model_weights.pth"
NUM_FIREFIGHTERS = 5
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

manager = None

# Per-firefighter locks so reads and writes never overlap
_file_locks = {i: threading.Lock() for i in range(NUM_FIREFIGHTERS)}

# In-memory state for each firefighter
state = {
    i: {"lat": 0.0, "lon": 0.0, "alt": 0.0, "label": None, "live": False, "image_ts": 0}
    for i in range(NUM_FIREFIGHTERS)
}


# ── Ingest endpoints (called by firefighter devices) ────────────────────────

@app.route("/api/image", methods=["POST"])
def receive_image():
    """Receive an image from a firefighter device.
    Accepts either:
      - multipart form: firefighter_id (int) + image (file)
      - raw JPEG bytes with Content-Type: image/jpeg (firefighter_id from query param)
    """
    if request.content_type and "multipart" in request.content_type:
        firefighter_id = int(request.form.get("firefighter_id", 0))
        data = request.files["image"].read()
    else:
        firefighter_id = int(request.args.get("firefighter_id", 0))
        data = request.data

    path = os.path.join(UPLOAD_DIR, f"firefighter_{firefighter_id}.jpg")
    with _file_locks[firefighter_id]:
        with open(path, "wb") as f:
            f.write(data)

    if manager:
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
    if manager:
        for i in range(NUM_FIREFIGHTERS):
            result = manager.get_result(worker_id=i)
            if result is not None:
                state[i]["label"] = result[1]

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
    path = os.path.join(UPLOAD_DIR, f"firefighter_{fid}.jpg")
    with _file_locks[fid]:
        return send_file(path, mimetype="image/jpeg")


if __name__ == "__main__":
    manager = FireFighterManager(MODEL_PATH, NUM_FIREFIGHTERS)
    app.run(host="0.0.0.0", debug=False)
