def mapResponse(similarityResults):
    return {
        "topics": {
            "similarity_results": similarityResults
        },
        "doc_topic": {},
        "metrics": {},
    }
