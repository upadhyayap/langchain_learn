from dotenv import load_dotenv
from ice_breaker import ice_break_with
from flask import Flask, request, jsonify, render_template

load_dotenv()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    name = request.form["name"]
    summary, photo_url = ice_break_with(name=name)
    return jsonify(
        {
            "summary": summary.summary,
            "facts": summary.facts,
            # "interests": summary.topics_of_interest,
            # "ice_breakers": summary.ice_breakers,
            "picture_url": photo_url,
        }
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)