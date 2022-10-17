from time import time
from flask import Flask, json, request

from src.feedUvlMapper import FeedUvlMapper
from src.techniques.vsm import UserStorySimilarityVsm
from src.techniques.wordnet import UserStorySimilarityWordnet
from src.techniques.word2vec import UserStorySimilarityWord2vec

app = Flask(__name__)

@app.route("/hitec/classify/concepts/us-similarity/run", methods=["POST"])
def post_user_stories():
    start = time()
    data = json.loads(request.data)
    mapper = FeedUvlMapper(app.logger)
    us_dataset, unextracted_us = mapper.map_request(data)
    is_focused, focused_ids = mapper.is_document_focused(data)
    threshold = mapper.get_threshold(data)

    technique = mapper.get_technique(data)
    us_similarity = None
    match technique:
        case "vsm":
            us_similarity = UserStorySimilarityVsm(mapper, threshold)
        case "wordnet":
            us_similarity = UserStorySimilarityWordnet(mapper, threshold)
        case "word2vec":
            us_similarity = UserStorySimilarityWord2vec(mapper, threshold)
        case _:
            pass

    unexistent_ids_count = 0
    result = []
    if is_focused:
        result, unexistent_ids_count = us_similarity.measure_pairwise_similarity(us_dataset, focused_ids, unextracted_us["ids"])
    else:
        result = us_similarity.measure_all_pairs_similarity(us_dataset)
    metrics = {
        "runtime": round(time() - start, 4),
        "user_story_count": len(us_dataset),
        "similar_us_pairs": len(result),
        "unextracted_us": unextracted_us["count"],
        "unexistent_ids": unexistent_ids_count
    }
    res = mapper.map_response(result, metrics)
            
    return res

@app.route("/hitec/classify/concepts/us-similarity/status", methods=["GET"])
def get_status():
    status = {"status": "operational"}
    return status

def before_start_up():
    UserStorySimilarityWord2vec.load_model(app.logger)

if __name__ == '__main__':  # started via: python3 app.py
    before_start_up()
    # app.run(debug=True)  # started via: flask (--debug) run
    app.run(debug=False, host="0.0.0.0", port=9697)
elif __name__ == 'app':
    before_start_up()

