from typing import Tuple
from src.exeptions import UserStoryParsingError

class FeedUvlMapper():

    def __init__(self, logger) -> None:
        self.logger = logger

    def map_request(self, req_data):
        docs = req_data["dataset"]["documents"]
        us_dataset = []
        for doc in docs:
            try:
                us_dataset.append({
                    "id": doc["id"],
                    "text": self.remove_newlines(self.extract_us(doc["text"])),
                    "acceptance_criteria": self.extract_acs(doc["text"]),
                    "raw_text": doc["text"]
                })
            except UserStoryParsingError:
                # user story could be recognized and will be skipped
                # TODO: include in metrics for feedUVL
                self.logger.warning(f'User story with id {doc["id"]} could not be extracted.')
                pass
        
        return us_dataset

    def extract_us(self, text: str) -> str:
        try:
            start = text.index("###")
            end = text.index("###", start+1)
            return text[start+3:end]
        except ValueError:
            raise UserStoryParsingError
    
    def extract_acs(self, text: str) -> str:
        try:
            start = text.index("+++")
            end = text.index("+++", start+1)
            return text[start+3:end]
        except ValueError:
            return ""

    def remove_newlines(self, doc_text: str) -> str:
        res = doc_text.replace("\n", " ")
        return res.strip()

    def is_document_focused(self, reqData) -> Tuple[bool, str]:
        params = reqData["params"]
        if not params["focused_document_id"]:
            return False, ""

        return True, params["focused_document_id"]

    def map_response(self, similarity_results_all_pairs, similarity_results_focused):
        return {
            "topics": {
                "similarity_results_all_pairs": similarity_results_all_pairs,
                "similarity_results_focused": similarity_results_focused
            },
            "doc_topic": None,
            "metrics": {},
            "codes": None
        }
