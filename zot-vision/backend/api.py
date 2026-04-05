import os
import base64
import tempfile
import importlib.util
import numpy as np
from flask import Flask, jsonify, request

# ─────────────────────────────────────────────────────────────
# WHY importlib instead of a normal "import threading"?
#
# Python has a built-in standard-library module also called
# "threading". If we just wrote  "from threading import ..."
# Python might grab the stdlib one instead of our local file.
# importlib lets us load our local threading.py by its exact
# file path, so there is no ambiguity.
# ─────────────────────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "firefighter_threading",                            # arbitrary internal name
    os.path.join(os.path.dirname(__file__), "threading.py"),  # path to our file
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)                         # actually runs threading.py
FireFighterManager = _mod.FireFighterManager           # grab the class we need

# ID_TO_LABEL maps the model's integer output → human-readable class name
# e.g.  0 → "human",  1 → "none",  2 → "both",  3 → "hazard"
from transformer import ID_TO_LABEL

app = Flask(__name__)

# Read config from environment variables so we don't have to edit code to change them.
# Falls back to sensible defaults if the env vars are not set.
MODEL_PATH       = os.environ.get("MODEL_PATH", "best_cnn_vit.pth")
NUM_FIREFIGHTERS = int(os.environ.get("NUM_FIREFIGHTERS", 4))

# One FireFighterManager is created at startup.
# Internally it spawns one worker process per firefighter, each with its own
# copy of the model loaded in memory — that's how we get parallel inference.
manager = FireFighterManager(MODEL_PATH, NUM_FIREFIGHTERS)

# Simple in-memory dict to hold the most recent GPS fix for each firefighter.
# Key   = firefighter_id (int)
# Value = {"lat": float, "lon": float}
gps_store = {}


# ─────────────────────────────────────────────────────────────
# ENDPOINT 1 — firmware pushes data TO the server
# ─────────────────────────────────────────────────────────────
@app.route("/handle_post1", methods=["POST"])
def handle_post1():
    """Receive image (base64) or GPS coordinates from firmware for one firefighter.

    Expected JSON body — image:
        { "firefighter_id": 0, "type": "image", "image": "<base64>", "ext": ".jpg" }

    Expected JSON body — GPS:
        { "firefighter_id": 0, "type": "gps", "lat": 33.64, "lon": -117.84 }
    """
    # Parse the incoming JSON body from the firmware request
    data           = request.get_json(force=True)
    firefighter_id = int(data.get("firefighter_id", 0))  # which firefighter sent this
    data_type      = data.get("type", "image")            # "image" or "gps"

    # ── GPS branch ──────────────────────────────────────────
    if data_type == "gps":
        # Just store the coordinates in our dict so handle_post2 can attach
        # them to the result later. No model inference needed for GPS.
        gps_store[firefighter_id] = {"lat": data["lat"], "lon": data["lon"]}
        return jsonify({"status": "ok", "firefighter_id": firefighter_id, "stored": "gps"})

    # ── Image branch ────────────────────────────────────────
    img_b64 = data.get("image")
    if not img_b64:
        # Firmware forgot to include the image — tell it with HTTP 400
        return jsonify({"error": "missing 'image' field"}), 400

    # The image arrives as a base64 string (text). Decode it back to raw bytes.
    img_bytes = base64.b64decode(img_b64)

    # Write those bytes to a temporary file on disk so the worker process
    # can open it with OpenCV / PIL (they need a file path, not raw bytes).
    # delete=False keeps the file alive after this block closes it;
    # we delete it manually in handle_post2 once inference is done.
    suffix = data.get("ext", ".jpg")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name  # e.g. /tmp/tmpXYZ.jpg

    # Enqueue the image for the correct worker — this call returns immediately
    # (non-blocking). The worker process will pick it up and run the model
    # in the background while the firmware can already send the next frame.
    manager.send_image(tmp_path, worker_id=firefighter_id)
    return jsonify({"status": "ok", "firefighter_id": firefighter_id, "stored": "image"})


# ─────────────────────────────────────────────────────────────
# ENDPOINT 2 — firmware (or dashboard) asks for the result
# ─────────────────────────────────────────────────────────────
@app.route("/handle_post2", methods=["POST"])
def handle_post2():
    """Return the latest model result for one firefighter (non-blocking poll).

    Expected JSON body:
        { "firefighter_id": 0 }

    Response when the result is ready:
        { "status": "ok", "firefighter_id": 0, "label": "human", "gps": {...} }

    Response when the worker is still processing:
        { "status": "pending", "firefighter_id": 0 }
    """
    data           = request.get_json(force=True)
    firefighter_id = int(data.get("firefighter_id", 0))

    # Check if the worker has finished processing the most recent image.
    # get_result() is non-blocking — returns None if nothing is ready yet.
    result = manager.get_result(worker_id=firefighter_id)

    if result is None:
        # Worker is still running inference. Tell the caller to try again later.
        return jsonify({"status": "pending", "firefighter_id": firefighter_id})

    # result is a tuple: (image_path, logits)
    # logits is a numpy array of shape (4,) — one raw score per class
    image_path, logits = result

    # argmax picks the index with the highest score → that's the predicted class
    pred_idx = int(np.argmax(logits))
    # Map index to string label, e.g. 0 → "human"
    label = ID_TO_LABEL[pred_idx]

    # The temp file is no longer needed — clean it up to avoid filling disk
    try:
        os.remove(image_path)
    except OSError:
        pass  # file might already be gone; that's fine

    # Return the classification label plus the last known GPS fix for this firefighter
    return jsonify({
        "status":         "ok",
        "firefighter_id": firefighter_id,
        "label":          label,                          # "human" | "none" | "both" | "hazard"
        "gps":            gps_store.get(firefighter_id),  # None if no GPS received yet
    })


if __name__ == "__main__":
    app.run(debug=False)
