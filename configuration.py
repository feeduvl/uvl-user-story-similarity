""" configuration module """

def get_supported_huggingface_models():
    return {
        "bert": "bert-large-cased",
        "roberta": "roberta-large",
        "roberta-stsb-v1": "sentence-transformers/stsb-roberta-large",
        "roberta-stsb-v2": "sentence-transformers/all-roberta-large-v1",
        "albert": "albert-base-v2",
        "bert4re": "thearod5/bert4re"
    }

def get_path_for_universal_sentence_encoder():
    # Chose between Online-Version and Cache-Version
    return "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    #return "<local directory>\\c9fe785512ca4a1b179831acb18a0c6bfba603dd"