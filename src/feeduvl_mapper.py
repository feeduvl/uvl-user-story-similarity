from typing import Tuple
from src.exeptions import UserStoryParsingError

class FeedUvlMapper():

    def __init__(self, logger) -> None:
        self.logger = logger

    def map_similarity_result(self, first, second, score, threshold, result):
        if score >= threshold:
            result_entry = {
                "id_1": first["id"],
                "id_2": second["id"],
                "us_1": first["text"],
                "us_2": second["text"],
                "score": round(score, 4),
                "ac_1": first["acceptance_criteria"],
                "ac_2": second["acceptance_criteria"],
                "raw_text_1": first["raw_text"],
                "raw_text_2": second["raw_text"]
            }
            result.append(result_entry)         

    def get_technique(self, req_data):
        return req_data["params"]["selected_technique"]

    def get_threshold(self, req_data):
        return float(req_data["params"]["threshold"])

    def map_request(self, req_data):
        docs = req_data["dataset"]["documents"]
        unextracted_us_count = 0
        unextracted_us_ids = []
        unextracted_ac_count = 0
        unextracted_ac_ids = []
        us_dataset = []
        for doc in docs:
            ac, is_ac_extracted = self._extract_acs(doc["text"])
            if not is_ac_extracted:
                self.logger.warning(f'Acceptance Criteria with US id {doc["id"]} could not be extracted.')
                unextracted_ac_count += 1
                unextracted_ac_ids.append(doc["id"])
            try:
                us_dataset.append({
                    "id": doc["id"],
                    "text": self._remove_newlines(self._extract_us(doc["text"])),
                    "acceptance_criteria": ac,
                    "raw_text": doc["text"]
                })
            except UserStoryParsingError:
                # user story could not be recognized and will be skipped
                self.logger.warning(f'User story with id {doc["id"]} could not be extracted.')
                unextracted_us_count += 1
                unextracted_us_ids.append(doc["id"])

        unextracted = {
            "us_count": unextracted_us_count,
            "us_ids": unextracted_us_ids,
            "ac_count": unextracted_ac_count,
            "ac_ids": unextracted_ac_ids
        }
        return us_dataset, unextracted
    

    def are_documents_focused(self, reqData) -> Tuple[bool, list[str]]:
        params = reqData["params"]
        if not params["focused_document_ids"]:
            return False, ""
        ids: str = params["focused_document_ids"]
        ids = ids.replace(" ", "")
        ids_array = ids.split(",")
        return True, ids_array

    def map_response(self, similarity_results, metrics):
        return {
            "topics": {
                "similarity_results": similarity_results
            },
            "doc_topic": None,
            "metrics": {
                "runtime_in_s": metrics["runtime"],
                "user_stories": metrics["user_story_count"],
                "similar_us_pairs": metrics["similar_us_pairs"],
                "unextracted_us": metrics["unextracted_us"],
                "unextracted_ac": metrics["unextracted_ac"],
                "unexistent_ids": metrics["unexistent_ids"]
            },
            "codes": None
        }

    def _extract_us(self, text: str) -> str:
        try:
            start = text.index("###")
            end = text.index("###", start+1)
            return text[start+3:end]
        except ValueError:
            raise UserStoryParsingError
    
    def _extract_acs(self, text: str) -> Tuple[str, bool]:
        try:
            start = text.index("+++")
            end = text.index("+++", start+1)
            return text[start+3:end], True
        except ValueError:
            return "", False

    def _remove_newlines(self, doc_text: str) -> str:
        res = doc_text.replace("\n", " ")
        return res.strip()
