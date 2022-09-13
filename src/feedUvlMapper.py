from src.exeptions import UserStoryParsingError

def mapRequest(reqData, logger):
    docs = reqData["dataset"]["documents"]
    us_dataset = []
    for doc in docs:
        try:
            us_dataset.append({
                "id": doc["id"],
                "text": remove_newlines(extract_us(doc["text"])),
                "acceptance_criteria": extract_acs(doc["text"]),
                "raw_text": doc["text"]
            })
        except UserStoryParsingError:
            # user story could be recognized and will be skipped
            # TODO: include in metrics for feedUVL
            logger.warning(f'User story with id {doc["id"]} could not be extracted.')
            pass
    return us_dataset

def extract_us(text: str) -> str:
    try:
        start = text.index("###")
        end = text.index("###", start+1)
        return text[start+3:end]
    except ValueError:
        raise UserStoryParsingError
    
def extract_acs(text: str) -> str:
    try:
        start = text.index("+++")
        end = text.index("+++", start+1)
        return text[start+3:end]
    except ValueError:
        return ""

def remove_newlines(doc_text: str) -> str:
     res = doc_text.replace("\n", " ")
     return res.strip()

def mapResponse(similarityResults):
    return {
        "topics": {
            "similarity_results": similarityResults
        },
        "doc_topic": {},
        "metrics": {},
    }
