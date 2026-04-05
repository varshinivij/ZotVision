import FireFighterManager
from flask import Flask, render_template, request, redirect, session

app = Flask(__name__)

def create_fire_fighter_manager(model_path="", num_firefighters=4):
    return FireFighterManager(model_path, num_firefighters)

@app.route('/handle_post1', methods=['POST'])
def handle_post():
    data = request.get_json()
    print("Received POST data:", data)
    return "POST request received"


@app.route('/handle_post2', methods=['POST'])
def handle_post2():
    data = request.get_json()
    print("Received POST data:", data)
    return "POST request received"

if __name__ == '__main__':
    app.run()