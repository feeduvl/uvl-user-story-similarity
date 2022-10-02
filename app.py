from flask import Flask, json, request

from src.feedUvlMapper import FeedUvlMapper
from src.techniques.vsm import UserStorySimilarityVsm
from src.techniques.wordnet import UserStorySimilarityWordnet
from src.techniques.word2vec import UserStorySimilarityWord2vec

app = Flask(__name__)

# TODO: name both endpoints right
# @app.route("/hitec/classify/concepts/lda/run", methods=["POST"])
@app.route("/run", methods=["POST"])
def post_user_stories():
    data = json.loads(request.data)
    mapper = FeedUvlMapper(app.logger)
    us_dataset = mapper.map_request(data)
    is_focused, focused_ids = mapper.is_document_focused(data)

    technique = mapper.get_technique(data)
    us_similarity = None
    match technique:
        case "vsm":
            us_similarity = UserStorySimilarityVsm(mapper)
        case "wordnet":
            us_similarity = UserStorySimilarityWordnet(mapper)
        case "word2vec":
            us_similarity = UserStorySimilarityWord2vec(mapper)
        case _:
            pass

    result = []
    if is_focused:
        result = us_similarity.measure_pairwise_similarity(us_dataset, focused_ids)
        res = mapper.map_response(result)
    else:
        result = us_similarity.measure_all_pairs_similarity(us_dataset)
        res = mapper.map_response(result)
            
    return res

@app.route("/status", methods=["GET"])
def get_status():
    status = {"status": "operational"}
    return status
