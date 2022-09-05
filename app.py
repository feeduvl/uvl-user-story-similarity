from flask import Flask, request, json

app = Flask(__name__)

# TODO: name both endpoints right
# @app.route("/hitec/classify/concepts/lda/run", methods=["POST"])
@app.route("/run", methods=["POST"])
def post_user_stories():
    data = json.loads(request.data)
    return data

@app.route("/status", methods=["GET"])
def get_status():
    status = {"status": "operational"}
    return status
