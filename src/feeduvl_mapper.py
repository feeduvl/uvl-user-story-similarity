""" feeduvl_mapper module """

import re
from typing import Tuple
from src.exeptions import UserStoryParsingError

class FeedUvlMapper():
    """ Mapper between Feed.UVL and this microservice """

    def __init__(self, logger) -> None:
        self.logger = logger

    def map_similarity_result(self, first, second, score, threshold, result):
        """ map a analysis run result to a similarity reponse object """
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

    def get_params(self, req_data):
        """ extract the parameter for the current request """
        remove_us_skeleton = False
        only_us_action = False
        no_preprocessing = False
        if "without_us_skeleton" in req_data["params"]:
            remove_us_skeleton = bool(req_data["params"]["without_us_skeleton"])
        if "only_us_action" in req_data["params"]:
            only_us_action = bool(req_data["params"]["only_us_action"])
        if "no_preprocessing" in req_data["params"]:
            no_preprocessing = bool(req_data["params"]["no_preprocessing"])
        focused_us_ids_list = []
        if "focused_document_ids" in req_data["params"]:
            focused_us_ids: str = req_data["params"]["focused_document_ids"]
            focused_us_ids = focused_us_ids.replace(" ", "")
            if focused_us_ids:
                focused_us_ids_list = focused_us_ids.split(",")

        if(no_preprocessing and (only_us_action or remove_us_skeleton)):
            raise """
                  Unsupported parameter set: if no_preprocessing is True, 
                  only_us_action and without_us_skeleton must be False.
                  """

        return {
            "technique": req_data["params"]["selected_technique"],
            "threshold": float(req_data["params"]["threshold"]),
            "are_us_focused": len(focused_us_ids_list) > 0,
            "focused_us_ids": focused_us_ids_list,
            "remove_us_skeleton": remove_us_skeleton,
            "only_us_action": only_us_action,
            "no_preprocessing": no_preprocessing
        }

    def map_request(self, req_data):
        """ map Feed.UVL request in internal data structure """
        docs = req_data["dataset"]["documents"]
        unextracted_us_count = 0
        unextracted_us_ids = []
        unextracted_ac_count = 0
        unextracted_ac_ids = []
        us_dataset = []
        avg_words = 0
        for doc in docs:
            ac, is_ac_extracted = self._extract_acs(doc["raw_text"])
            if not is_ac_extracted:
                self.logger.warning(f'Acceptance Criteria with US id {doc["id"]} could not be extracted.')
                unextracted_ac_count += 1
                unextracted_ac_ids.append(doc["id"])
            try:
                us_text = self._extract_us(doc["raw_text"])
                us_dataset.append({
                    "id": doc["id"],
                    "text": us_text,
                    "acceptance_criteria": ac,
                    "raw_text": doc["text"]
                })
                avg_words += self._get_word_count(us_text)
            except UserStoryParsingError:
                # user story could not be recognized and will be skipped
                self.logger.warning(f'User story with id {doc["id"]} could not be extracted.')
                unextracted_us_count += 1
                unextracted_us_ids.append(doc["id"])

        len_us_dataset = 1 if len(us_dataset) == 0 else len(us_dataset)
        avg_words /= len_us_dataset
        metrics = {
            "us_count": unextracted_us_count,
            "us_ids": unextracted_us_ids,
            "ac_count": unextracted_ac_count,
            "ac_ids": unextracted_ac_ids,
            "avg_words": avg_words
        }
        return us_dataset, metrics

    def map_response(self, similarity_results, metrics):
        """ Map the result of the analysis run into a Feed.UVL response """
        return {
            "topics": {
                "similarity_results": similarity_results
            },
            "doc_topic": None,
            "metrics": metrics,
            "codes": None
        }

    def _extract_us(self, text: str) -> str:
        try:
            start = text.index("###")
            end = text.index("###", start+1)
            us = text[start+3:end]
            return self._remove_newlines(us)
        except ValueError as value_err:
            raise UserStoryParsingError from value_err

    def _extract_acs(self, text: str) -> Tuple[str, bool]:
        try:
            start = text.index("+++")
            end = text.index("+++", start+1)
            return text[start+3:end].strip(), True
        except ValueError:
            return "", False

    def _remove_newlines(self, doc_text: str) -> str:
        res = doc_text.replace("\n", " ")
        return res.strip()

    def _get_word_count(self, us_text: str) -> int:
        return len(re.findall(r'\w+', us_text))
