from typing import Tuple
from src.exeptions import UserStoryParsingError

class FeedUvlMapper():

    def __init__(self, logger) -> None:
        self.logger = logger

    def map_to_us_representation(self, first, second, score, result, threshold):
        if score >= threshold:
            result_entry = {
                "id_1": first["id"],
                "id_2": second["id"],
                "us_1": first["text"],
                "us_2": second["text"],
                "score": score,
                "ac_1": first["acceptance_criteria"],
                "ac_2": second["acceptance_criteria"],
                "raw_text_1": first["raw_text"],
                "raw_text_2": second["raw_text"]
            }
            result.append(result_entry)         

    def get_technique(self, req_data):
        return req_data["params"]["selected_technique"]

    def get_threshold(self, req_data):
        return req_data["params"]["threshold"]

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

    def is_document_focused(self, reqData) -> Tuple[bool, list[str]]:
        params = reqData["params"]
        if not params["focused_document_ids"]:
            return False, ""
        ids: str = params["focused_document_ids"]
        ids = ids.replace(" ", "")
        ids_array = ids.split(",")
        return True, ids_array

    def map_response(self, similarity_results):
        return {
            "topics": {
                "similarity_results": similarity_results
            },
            "doc_topic": None,
            "metrics": {},
            "codes": None
        }
