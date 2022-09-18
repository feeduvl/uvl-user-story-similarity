from flask import Flask, json, request

from src.feedUvlMapper import FeedUvlMapper
from src.techniques.vsm import UserStorySimilarityVsm

app = Flask(__name__)

# TODO: name both endpoints right
# @app.route("/hitec/classify/concepts/lda/run", methods=["POST"])
@app.route("/run", methods=["POST"])
def post_user_stories():
    data = json.loads(request.data)
    mapper = FeedUvlMapper(app.logger)
    us_dataset = mapper.map_request(data)
    is_focused, focused_id = mapper.is_document_focused(data)

    vsm_similarity = UserStorySimilarityVsm()
    result = []
    res = {}
    if is_focused:
        result = vsm_similarity.measure_pairwise_similarity(us_dataset, focused_id)
        res = mapper.map_response([], result)
    else:
        result = vsm_similarity.measure_all_pairs_similarity(us_dataset)
        res = mapper.map_response(result, {})
            
    return res

@app.route("/status", methods=["GET"])
def get_status():
    status = {"status": "operational"}
    return status
