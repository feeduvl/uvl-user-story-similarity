from flask import Flask, json, request

from src.feedUvlMapper import mapRequest, mapResponse, is_document_focused
from src.techniques.vsm import UserStorySimilarityVsm

app = Flask(__name__)

# TODO: name both endpoints right
# @app.route("/hitec/classify/concepts/lda/run", methods=["POST"])
@app.route("/run", methods=["POST"])
def post_user_stories():
    data = json.loads(request.data)
    us_dataset = mapRequest(data, app.logger)
    is_focused, focused_id = is_document_focused(data)

    vsmSimilarity = UserStorySimilarityVsm()
    result = []
    res = {}
    if is_focused:
        result = vsmSimilarity.measure_pairwise_similarity(us_dataset, focused_id)
        res = mapResponse(data, [], result)
    else:
        result = vsmSimilarity.measure_all_pairs_similarity(us_dataset)
        res = mapResponse(data, result, {})
    
    return res

@app.route("/status", methods=["GET"])
def get_status():
    status = {"status": "operational"}
    return status
