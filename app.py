from flask import Flask, jsonify
from flask_cors import CORS
import recommendation

app = Flask(__name__)
CORS(app)


@app.route('/menu/<title>', methods=['GET'])
def recommend_food(title: str):
    res = recommendation.results(title)
    return jsonify(res)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
