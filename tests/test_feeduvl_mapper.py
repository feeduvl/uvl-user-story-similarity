from src.feeduvl_mapper import FeedUvlMapper
from unittest import TestCase
import pytest
import copy
import logging

user_story_1 = {
    "id": "COMET-1",
    "text": "Example US 1",
    "acceptance_criteria": "Example AC 1",
    "raw_text": "Example Raw Text 1"
}

user_story_2 = {
    "id": "COMET-2",
    "text": "Example US 2",
    "acceptance_criteria": "Example AC 2",
    "raw_text": "Example Raw Text 2"
}

expected_result_dict = {
    "id_1": "COMET-1",
    "id_2": "COMET-2",
    "us_1": "Example US 1",
    "us_2": "Example US 2",
    "score": 0.8444,
    "ac_1": "Example AC 1",
    "ac_2": "Example AC 2",
    "raw_text_1": "Example Raw Text 1",
    "raw_text_2": "Example Raw Text 2"
}

req_data = {
    "method": "TestMethod",
    "params": {
        "focused_document_ids": "",
        "name": "",
        "selected_technique": "vsm",
        "threshold": 0.7
    },
    "dataset": {
        "name": "test-data-set",
        "documents": [
            {
                "id": "COMET-203",
                "number": 0,
                "text": "###\nAs an user\nI want to do sth.,\nso that I can achieve sth.\n###\n\nAcceptance criteria (post conditions):\n+++* AC 1+++"
            },
            {
                "id": "COMET-204",
                "number": 0,
                "text": "###\nAs an user\nI want to do sth.,\nso that I can achieve sth.\n###\n\nAcceptance criteria (post conditions):\n+++* AC 1+++"
            }
        ],
        "ground_truth": []
    }
}

expected_us_dataset_entry_dict = {
    "id": "COMET-203",
    "text": "As an user I want to do sth., so that I can achieve sth.",
    "acceptance_criteria": "* AC 1",
    "raw_text": "###\nAs an user\nI want to do sth.,\nso that I can achieve sth.\n###\n\nAcceptance criteria (post conditions):\n+++* AC 1+++"
}

similarity_results = [
    {
        "id_1": "COMET-1",
        "id_2": "COMET-2",
        "us_1": "Example US 1",
        "us_2": "Example US 2",
        "score": 0.8444,
        "ac_1": "Example AC 1",
        "ac_2": "Example AC 2",
        "raw_text_1": "Example Raw Text 1",
        "raw_text_2": "Example Raw Text 2"
    }
]

result_metrics = {
    "runtime": 0.2,
    "user_story_count": 2,
    "similar_us_pairs": 1,
    "unextracted_us": 0,
    "unexistent_ids": 0
}

expected_response = {
    "topics": {
        "similarity_results": similarity_results
    },
    "doc_topic": None,
    "metrics": {
        "runtime_in_s": result_metrics["runtime"],
        "user_stories": result_metrics["user_story_count"],
        "similar_us_pairs": result_metrics["similar_us_pairs"],
        "unextracted_us": result_metrics["unextracted_us"],
        "unexistent_ids": result_metrics["unexistent_ids"]
    },
    "codes": None
}

@pytest.fixture
def mapper():
    logger = logging.getLogger("testLogger")
    return FeedUvlMapper(logger)

# test map_similarity_result method
@pytest.mark.parametrize("first_us,second_us", [(user_story_1, user_story_2)])
def test_map_similarity_result_not_similar(mapper: FeedUvlMapper, first_us, second_us):
    result = []
    mapper.map_similarity_result(first_us, second_us, 0.2, 0.7, result)
    assert not result

@pytest.mark.parametrize("first_us,second_us,expected_result_dict", [(user_story_1, user_story_2, expected_result_dict)])
def test_map_similarity_result_similar(mapper: FeedUvlMapper, first_us, second_us, expected_result_dict):
    result = []
    mapper.map_similarity_result(first_us, second_us, 0.844444, 0.7, result)
    assert result
    TestCase().assertDictEqual(expected_result_dict, result[0])

@pytest.mark.parametrize("first_us,second_us,expected_result_dict", [(user_story_1, user_story_2, expected_result_dict)])
def test_map_similarity_result_similar_rounded_up(mapper: FeedUvlMapper, first_us, second_us, expected_result_dict):
    expected_result_dict_with_rounded_up_score = copy.deepcopy(expected_result_dict)
    expected_result_dict_with_rounded_up_score["score"] = 0.8556

    result = []
    mapper.map_similarity_result(first_us, second_us, 0.855555, 0.7, result)
    assert result
    TestCase().assertDictEqual(expected_result_dict_with_rounded_up_score, result[0])

# test get_technique method
@pytest.mark.parametrize("req_data", [req_data])
def test_get_technique(mapper: FeedUvlMapper, req_data):
    technique = mapper.get_technique(req_data)
    assert technique == req_data["params"]["selected_technique"]

@pytest.mark.parametrize("req_data", [req_data])
def test_get_technique_missing_param(mapper: FeedUvlMapper, req_data):
    req_data_temp = copy.deepcopy(req_data)
    del req_data_temp["params"]["selected_technique"]
    with pytest.raises(Exception) as e_info:
        mapper.get_technique(req_data_temp)

# test get_threshold method
@pytest.mark.parametrize("req_data", [req_data])
def test_get_threshold(mapper: FeedUvlMapper, req_data):
    threshold = mapper.get_threshold(req_data)
    assert threshold == req_data["params"]["threshold"]
    assert isinstance(threshold, float)

@pytest.mark.parametrize("req_data", [req_data])
def test_get_threshold_casts_to_float(mapper: FeedUvlMapper, req_data):
    temp_threshold = "0.7"
    req_data_temp = copy.deepcopy(req_data)
    req_data_temp["params"]["threshold"] = temp_threshold
    threshold = mapper.get_threshold(req_data_temp)
    assert threshold == float(temp_threshold)
    assert isinstance(threshold, float)

# test map_request method
@pytest.mark.parametrize("req_data,expected_us_dataset_entry_dict", [(req_data, expected_us_dataset_entry_dict)])
def test_map_request(mapper: FeedUvlMapper, req_data, expected_us_dataset_entry_dict):
    us_dataset, unextracted_us = mapper.map_request(req_data)
    assert unextracted_us["count"] == 0
    assert not unextracted_us["ids"]
    assert len(us_dataset) == 2
    TestCase().assertDictEqual(expected_us_dataset_entry_dict, us_dataset[0])

@pytest.mark.parametrize("req_data", [req_data])
def test_map_request_unextracted_us(mapper: FeedUvlMapper, req_data):
    req_data_temp = copy.deepcopy(req_data)
    req_data_temp["dataset"]["documents"][0]["text"] = "##\nAs an user\nI want to do sth.,\nso that I can achieve sth.\n###\n\nAcceptance criteria (post conditions):\n+++* AC 1+++"
    us_dataset, unextracted_us = mapper.map_request(req_data_temp)
    assert unextracted_us["count"] == 1
    assert len(unextracted_us["ids"]) == 1
    assert unextracted_us["ids"][0] == req_data_temp["dataset"]["documents"][0]["id"]
    assert len(us_dataset) == 1

@pytest.mark.parametrize("req_data", [req_data])
def test_map_request_unextracted_ac(mapper: FeedUvlMapper, req_data):
    wrong_raw_text = "###\nAs an user\nI want to do sth.,\nso that I can achieve sth.\n###\n\nAcceptance criteria (post conditions):\n++* AC 1+++"
    expected_us_dataset_entry_dict_temp = copy.deepcopy(expected_us_dataset_entry_dict)
    expected_us_dataset_entry_dict_temp["acceptance_criteria"] = ""
    expected_us_dataset_entry_dict_temp["raw_text"] = wrong_raw_text
    req_data_temp = copy.deepcopy(req_data)
    req_data_temp["dataset"]["documents"][0]["text"] = wrong_raw_text
    us_dataset, unextracted_us = mapper.map_request(req_data_temp)
    assert unextracted_us["count"] == 0
    assert not unextracted_us["ids"]
    TestCase().assertDictEqual(expected_us_dataset_entry_dict_temp, us_dataset[0])

# test are_documents_focused method
@pytest.mark.parametrize("req_data", [req_data])
def test_is_docment_focused_false(mapper: FeedUvlMapper, req_data):
    focused, ids = mapper.are_documents_focused(req_data)
    assert focused == False
    assert not ids

@pytest.mark.parametrize("req_data", [req_data])
def test_is_docment_focused_true(mapper: FeedUvlMapper, req_data):
    req_data_temp = copy.deepcopy(req_data)
    req_data_temp["params"]["focused_document_ids"] = "COMET-1, COMET-2"
    focused, ids = mapper.are_documents_focused(req_data_temp)
    assert focused == True
    assert len(ids) == 2
    assert ids[0] == "COMET-1"
    assert ids[1] == "COMET-2"

# test map_response method
@pytest.mark.parametrize("similarity_results,result_metrics,expected_response", [(similarity_results, result_metrics, expected_response)])
def test_map_response(mapper: FeedUvlMapper, similarity_results, result_metrics, expected_response):
    response = mapper.map_response(similarity_results, result_metrics)
    TestCase().assertDictEqual(expected_response, response)
