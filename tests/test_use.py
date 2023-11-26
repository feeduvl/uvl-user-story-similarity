from src.techniques.use import UserStorySimilarityUse
from src.feeduvl_mapper import FeedUvlMapper
import pytest
import logging

us_dataset = [
    {
        "id": "TEST-1",
        "text": "As a system tester I want to know which user stories are similar, so that I remove inconsistencies.",
        "acceptance_criteria": "* Identical user stories have a maximum score.\n* Non-identical user stories do not have a maximum score.",
        "raw_text": "###\nAs a system tester I want to know which user stories are similar, so that I remove inconsistencies.\n###\n\nAcceptance criteria (postconditions):\n+++\n* Identical user stories have a maximum score.\n* Non-identical user stories do not have a maximum score.\n+++"
    },
    {
        "id": "TEST-2",
        "text": "As a software engineer I want to know which user stories are similar, so that I remove inconsistencies.",
        "acceptance_criteria": "* Identical user stories have a maximum score.\n* Non-identical user stories do not have a maximum score.",
        "raw_text": "###\nAs a software engineer I want to know which user stories are similar, so that I remove inconsistencies.\n###\n\nAcceptance criteria (postconditions):\n+++\n* Identical user stories have a maximum score.\n* Non-identical user stories do not have a maximum score.\n+++"
    },
    {
        "id": "TEST-3",
        "text": "As a cashier I want to know how much money the customer needs to pay, so that I do not have to make the calculations.",
        "acceptance_criteria": "* A display shows the amount to be paid.\n",
        "raw_text": "###\nAs a cashier I want to know how much money the customer needs to pay, so that I do not have to make the calculations.\n###\n\nAcceptance criteria (postconditions):\n+++\n* A display shows the amount to be paid.\n+++"
    }
]

@pytest.fixture
def usSimilarityUse():
    logger = logging.getLogger("testLogger")
    mapper = FeedUvlMapper(logger) # mock this
    return UserStorySimilarityUse(mapper, 0.1, True, False, False)

# test measure_all_pairs_similarity method
@pytest.mark.parametrize("us_dataset", [(us_dataset)])
def test_measure_all_pairs_similarity(usSimilarityUse: UserStorySimilarityUse, us_dataset):
    result = usSimilarityUse.measure_all_pairs_similarity(us_dataset)
    assert len(result) == 3
    # Test-1 and Test-2 at index 0 are pretty similar, 
    # so they should have a greater score than Test-1 and Test-3 at index 2
    assert result[0]["score"] > result[1]["score"]

def test_measure_all_pairs_similarity_one_us(usSimilarityUse: UserStorySimilarityUse):
    result = usSimilarityUse.measure_all_pairs_similarity(["I do not matter"])
    assert result == []

def test_measure_all_pairs_similarity_no_us(usSimilarityUse: UserStorySimilarityUse):
    result = usSimilarityUse.measure_all_pairs_similarity([])
    assert result == []

# test measure_pairwise_similarity method
@pytest.mark.parametrize("us_dataset", [(us_dataset)])
def test_measure_pairwise_similarity(usSimilarityUse: UserStorySimilarityUse, us_dataset):
    result, unexistent_ids = usSimilarityUse.measure_pairwise_similarity(us_dataset, ["TEST-1"], [])
    assert len(result) == 2
    assert unexistent_ids == 0
    assert result[0]

def test_measure_pairwise_similarity_one_us(usSimilarityUse: UserStorySimilarityUse):
    result, unexistent_ids = usSimilarityUse.measure_pairwise_similarity(["I do not matter"], ["TEST-1"], [])
    assert result == []
    assert unexistent_ids == 0

def test_measure_pairwise_similarity_no_us(usSimilarityUse: UserStorySimilarityUse):
    result, unexistent_ids = usSimilarityUse.measure_pairwise_similarity([], ["TEST-1"], [])
    assert result == []
    assert unexistent_ids == 0

@pytest.mark.parametrize("us_dataset", [us_dataset])
def test_measure_pairwise_similarity_unexistent_id(usSimilarityUse: UserStorySimilarityUse, us_dataset):
    result, unexistent_ids = usSimilarityUse.measure_pairwise_similarity(us_dataset, ["Unexistent-1"], [])
    assert len(result) == 0
    assert unexistent_ids == 1

@pytest.mark.parametrize("us_dataset", [us_dataset])
def test_measure_pairwise_similarity_unexistent_id_not_extracted(usSimilarityUse: UserStorySimilarityUse, us_dataset):
    result, unexistent_ids = usSimilarityUse.measure_pairwise_similarity(us_dataset, ["Unexistent-1"], ["Unexistent-1"])
    assert len(result) == 0
    assert unexistent_ids == 0
