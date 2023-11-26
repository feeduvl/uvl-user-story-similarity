from time import time
from flask import Flask, json, request

from src.feeduvl_mapper import FeedUvlMapper
from src.techniques.vsm import UserStorySimilarityVsm
from src.techniques.wordnet import UserStorySimilarityWordnet
from src.techniques.word2vec import UserStorySimilarityWord2vec

app = Flask(__name__)

@app.route("/hitec/classify/concepts/us-similarity/run", methods=["POST"])
def start_analysis():
    start = time()
    data = json.loads(request.data)
    mapper = FeedUvlMapper(app.logger)
    us_dataset, dataset_metrics = mapper.map_request(data)

    params = mapper.get_params(data)

    us_similarity = None
    technique = params["technique"]
    if(technique == "vsm"):
        us_similarity = UserStorySimilarityVsm(
            mapper,
            params["threshold"],
            params["remove_us_skeleton"],
            params["only_us_action"]
        )
    elif(technique == "wordnet"):
        us_similarity = UserStorySimilarityWordnet(
            mapper,
            params["threshold"],
            params["remove_us_skeleton"],
            params["only_us_action"]
        )
    elif(technique == "word2vec"):
        us_similarity = UserStorySimilarityWord2vec(
            mapper,
            params["threshold"],
            params["remove_us_skeleton"],
            params["only_us_action"]
        )
    elif(technique in supported_huggingface_models):
        us_similarity = UserStorySimilarityBertBased(
            mapper,
            params["threshold"],
            params["remove_us_skeleton"],
            params["only_us_action"],
            params["no_preprocessing"],
            supported_huggingface_models[technique]
        )
    elif(technique == "use"):
        us_similarity = UserStorySimilarityUse(
            mapper,
            params["threshold"],
            params["remove_us_skeleton"],
            params["only_us_action"],
            params["no_preprocessing"]
        )
    else:
        raise f"Unsupported technique: {technique}"

    unexistent_ids_count = 0
    result = []
    if params["are_us_focused"]:
        result, unexistent_ids_count = us_similarity.measure_pairwise_similarity(us_dataset, params["focused_us_ids"], dataset_metrics["us_ids"])
    else:
        result = us_similarity.measure_all_pairs_similarity(us_dataset)
    metrics = {
        "runtime_in_s": round(time() - start, 4),
        "user_stories": len(us_dataset),
        "similar_us_pairs": len(result),
        "unextracted_us": dataset_metrics["us_count"],
        "unextracted_ac": dataset_metrics["ac_count"],
        "unexistent_ids": unexistent_ids_count,
        "avg_words": dataset_metrics["avg_words"]
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

