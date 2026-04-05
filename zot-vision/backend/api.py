import FireFighterManager
from flask import Flask, render_template, request, redirect, session

app = Flask(__name__)

def create_fire_fighter_manager(model_path="", num_firefighters=4):
    return FireFighterManager(model_path, num_firefighters)


@app.route('/handle_get', methods=['GET'])

@app.route('/handle_post', methods=['POST'])

if __name__ == '__main__':
    app.run()