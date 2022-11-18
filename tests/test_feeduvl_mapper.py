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
        "threshold": 0.7,
        "without_us_skeleton": False,
        "only_us_action": False
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

expected_us_dataset_entry_dict_croped_us = {
    "id": "COMET-203",
    "text": "As an user I want to do sth., so that I can",
    "acceptance_criteria": "* AC 1",
    "raw_text": "###\nAs an user\nI want to do sth.,\nso that I can### achieve sth.\n###\n\nAcceptance criteria (post conditions):\n+++* AC 1+++"
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
    "runtime_in_s": 0.2,
    "user_stories": 2,
    "similar_us_pairs": 1,
    "unextracted_us": 0,
    "unextracted_ac": 0,
    "unexistent_ids": 0,
    "avg_words": 3
}

expected_response = {
    "topics": {
        "similarity_results": similarity_results
    },
    "doc_topic": None,
    "metrics": result_metrics,
    "codes": None
}

expected_params = {
    "threshold": 0.7,
    "technique": "vsm",
    "are_us_focused": False,
    "focused_us_ids": [],
    "remove_us_skeleton": False,
    "only_us_action": False
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

# test get_params method
@pytest.mark.parametrize("req_data,expected_params", [(req_data, expected_params)])
def test_get_params(mapper: FeedUvlMapper, req_data, expected_params):
    params = mapper.get_params(req_data)
    TestCase().assertDictEqual(expected_params, params)

# test map_request method
@pytest.mark.parametrize("req_data,expected_us_dataset_entry_dict", [(req_data, expected_us_dataset_entry_dict)])
def test_map_request(mapper: FeedUvlMapper, req_data, expected_us_dataset_entry_dict):
    us_dataset, unextracted = mapper.map_request(req_data)
    assert unextracted["us_count"] == 0
    assert not unextracted["us_ids"]
    assert unextracted["ac_count"] == 0
    assert not unextracted["ac_ids"]
    assert len(us_dataset) == 2
    TestCase().assertDictEqual(expected_us_dataset_entry_dict, us_dataset[0])

@pytest.mark.parametrize("req_data", [req_data])
def test_map_request_unextracted_us(mapper: FeedUvlMapper, req_data):
    req_data_temp = copy.deepcopy(req_data)
    req_data_temp["dataset"]["documents"][0]["text"] = "##\nAs an user\nI want to do sth.,\nso that I can achieve sth.\n###\n\nAcceptance criteria (post conditions):\n+++* AC 1+++"
    us_dataset, unextracted = mapper.map_request(req_data_temp)
    assert unextracted["us_count"] == 1
    assert len(unextracted["us_ids"]) == 1
    assert unextracted["us_ids"][0] == req_data_temp["dataset"]["documents"][0]["id"]
    assert unextracted["ac_count"] == 0
    assert not unextracted["ac_ids"]
    assert len(us_dataset) == 1

@pytest.mark.parametrize("req_data,expected_us_dataset_entry_dict_croped_us", [(req_data, expected_us_dataset_entry_dict_croped_us)])
def test_map_request_more_delimiters_than_needed(mapper: FeedUvlMapper, req_data, expected_us_dataset_entry_dict_croped_us):
    req_data_temp = copy.deepcopy(req_data)
    req_data_temp["dataset"]["documents"][0]["text"] = "###\nAs an user\nI want to do sth.,\nso that I can### achieve sth.\n###\n\nAcceptance criteria (post conditions):\n+++* AC 1+++"
    us_dataset, unextracted = mapper.map_request(req_data_temp)
    assert unextracted["us_count"] == 0
    assert not unextracted["us_ids"]
    assert unextracted["ac_count"] == 0
    assert not unextracted["ac_ids"]
    assert len(us_dataset) == 2
    TestCase().assertDictEqual(expected_us_dataset_entry_dict_croped_us, us_dataset[0])

@pytest.mark.parametrize("req_data", [req_data])
def test_map_request_unextracted_ac(mapper: FeedUvlMapper, req_data):
    wrong_raw_text = "###\nAs an user\nI want to do sth.,\nso that I can achieve sth.\n###\n\nAcceptance criteria (post conditions):\n++* AC 1+++"
    expected_us_dataset_entry_dict_temp = copy.deepcopy(expected_us_dataset_entry_dict)
    expected_us_dataset_entry_dict_temp["acceptance_criteria"] = ""
    expected_us_dataset_entry_dict_temp["raw_text"] = wrong_raw_text
    req_data_temp = copy.deepcopy(req_data)
    req_data_temp["dataset"]["documents"][0]["text"] = wrong_raw_text
    us_dataset, unextracted = mapper.map_request(req_data_temp)
    assert unextracted["us_count"] == 0
    assert not unextracted["us_ids"]
    assert unextracted["ac_count"] == 1
    assert unextracted["ac_ids"][0] == req_data_temp["dataset"]["documents"][0]["id"]
    TestCase().assertDictEqual(expected_us_dataset_entry_dict_temp, us_dataset[0])

# test map_response method
@pytest.mark.parametrize("similarity_results,result_metrics,expected_response", [(similarity_results, result_metrics, expected_response)])
def test_map_response(mapper: FeedUvlMapper, similarity_results, result_metrics, expected_response):
    response = mapper.map_response(similarity_results, result_metrics)
    TestCase().assertDictEqual(expected_response, response)
