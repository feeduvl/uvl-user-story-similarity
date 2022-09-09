from flask import Flask, json, request
from src.feedUvlMapper import mapResponse

from src.techniques.vsm import UserStorySimilarityVsm
from src.mock.mockData import microserviceDataRepresentation

app = Flask(__name__)

# TODO: name both endpoints right
# @app.route("/hitec/classify/concepts/lda/run", methods=["POST"])
@app.route("/run", methods=["POST"])
def post_user_stories():
    # data = json.loads(request.data)
    vsmSimilarity = UserStorySimilarityVsm()
    result = vsmSimilarity.measure_similarity(microserviceDataRepresentation)  # TODO: Use real request data
    res = mapResponse(result)

    return res

@app.route("/status", methods=["GET"])
def get_status():
    status = {"status": "operational"}
    return status
