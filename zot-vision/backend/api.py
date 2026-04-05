import FireFighterManager
from flask import Flask, render_template, request, redirect, session, jsonify

app = Flask(__name__)

MODEL_PATH = "../datasets/results/model_weights.pth"
NUM_FIREFIGHTERS = 5

manager = FireFighterManager(MODEL_PATH, NUM_FIREFIGHTERS)


@app.route('/handle_post1', methods=['POST'])
def handle_post():
    data = request.get_json(force=True)
    firefighter_id = int(data.get("firefighter_id", 0))
    result = manager.get_result(worker_id=firefighter_id)
    if result is None:
        # Worker is still running inference. Tell the caller to try again later.
        return jsonify({"status": "pending", "firefighter_id": firefighter_id})

    #send this to App.jsx to display it on the frontend
    return jsonify({
        "status":         "ok",
        "firefighter_id": firefighter_id,
        "label":          result
    })

"""
Expected JSON body — GPS:
{ "firefighter_id": 0, "type": "gps", "lat": 33.64, "lon": -117.84 }
"""
@app.route('/handle_post2', methods=['POST'])
def handle_post2():
    data = request.get_json(force=True)
    firefighter_id = int(data.get("firefighter_id", 0))
    lat = float(data.get("lat", 0.0))
    lon = float(data.get("lon", 0.0))
    alt = float(data.get("alt", 0.0))
    #send this to App.jsx to display it on the frontend
    return jsonify({
        "status":         "ok",
        "firefighter_id": firefighter_id,
        "lat":            lat,
        "lon":            lon,
        "alt":            alt
    })


if __name__ == "__main__":
    app.run(debug=False)