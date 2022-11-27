from src.techniques.vsm import UserStorySimilarityVsm
from src.feeduvl_mapper import FeedUvlMapper
from unittest import TestCase
import pytest
import logging

us_dataset = [
    {
        "id": "COMET-183",
        "text": "As an Event Manager I want to be able to edit a contact in the contact details view and unsubscribe him manually from an email distribution list, so that I can process unsubscribe requests via other channels (e.g. phone).",
        "acceptance_criteria": "* In the event details with the matching distribution list, the contact is no longer on the distribution list.\n* A corresponding Incident can be found in the contact history.",
        "raw_text": "###\nAs an Event Manager\nI want to be able to edit a contact in the contact details view and unsubscribe him manually from an email distribution list,\nso that I can process unsubscribe requests via other channels (e.g. phone).\n###\nThe contact should get an Incident (with manual description of the reason).\nStorage period: 6 weeks\n\nAcceptance criteria (postconditions):\n+++\n* In the event details with the matching distribution list, the contact is no longer on the distribution list.\n* A corresponding Incident can be found in the contact history.\n+++"
    },
    {
        "id": "COMET-224",
        "text": "As an Event Manager I want that in the view Event Details -> Feedback by importing a CSV file, all contacts with positive feedback are automatically put on all distribution lists of the location belonging to the event, so that all participants with positive feedback are informed in the future about all events at the given location.",
        "acceptance_criteria": "1. all contacts with positive feedback are on all distribution lists that belong to the same location as the event associated with the feedback.\n2. all contacts with positive feedback have the corresponding incidents in their contact history.",
        "raw_text": "###\nAs an Event Manager\nI want that in the view Event Details -> Feedback by importing a CSV file, all contacts with positive feedback are automatically put on all distribution lists of the location belonging to the event, so that all participants with positive feedback are informed in the future about all events at the given location.\n###\nExample: Mr. P. Müller visits the OF Karlsruhe and clicks on the link \"Continue to be informed about events from andrena\" in the thank you email after participation. After importing the feedback, the contact P. Müller should be registered for the distribution list of all events in Karlsruhe (OF, Entwicklertage, XPDays, Workshops) and not only for the OF.\n\nHint:\nPositive feedback means: Clicked=Yes, Deregistered=No.\n----\nAcceptance Criteria:\n+++\n1. all contacts with positive feedback are on all distribution lists that belong to the same location as the event associated with the feedback.\n2. all contacts with positive feedback have the corresponding incidents in their contact history.\n+++"
    },
    {
        "id": "COMET-225",
        "text": "As an event manager I want that in the Event Details -> Feedback view, by importing a CSV file, all contacts with negative feedback are automatically removed from all distribution lists, so that these contacts do not accidentally receive an email from andrena.",
        "acceptance_criteria": "1. all contacts with negative feedback are not subscribed to any distribution list.\n2. all contacts with negative feedback have the corresponding incidents in their history. The consequence for the deletion period is that the contact has only 14 days storage period and the previously given permanent storage period is canceled.",
        "raw_text": "###\nAs an event manager\nI want that in the Event Details -> Feedback view, by importing a CSV file, all contacts with negative feedback are automatically removed from all distribution lists, so that these contacts do not accidentally receive an email from andrena.\n###\n_Example: Mr. P. Müller attends the OF Karlsruhe and clicks on the link \"Do not receive any more information\" in the thank you email after attending. After importing the feedback, the contact P. Müller should be removed from all distribution lists._\n\nNotice:\nPositive feedback means: Clicked=Yes, Unsubscribed=No.\n----\nAcceptance Criteria:\n+++\n1. all contacts with negative feedback are not subscribed to any distribution list.\n2. all contacts with negative feedback have the corresponding incidents in their history. The consequence for the deletion period is that the contact has only 14 days storage period and the previously given permanent storage period is canceled.\n+++"
    }
]

expected_similarity_result_entry_1 = {
    "ac_1": "* In the event details with the matching distribution list, the contact is no longer on the distribution list.\n* A corresponding Incident can be found in the contact history.",
    "ac_2": "1. all contacts with positive feedback are on all distribution lists that belong to the same location as the event associated with the feedback.\n2. all contacts with positive feedback have the corresponding incidents in their contact history.",
    "id_1": "COMET-183",
    "id_2": "COMET-224",
    "raw_text_1": "###\nAs an Event Manager\nI want to be able to edit a contact in the contact details view and unsubscribe him manually from an email distribution list,\nso that I can process unsubscribe requests via other channels (e.g. phone).\n###\nThe contact should get an Incident (with manual description of the reason).\nStorage period: 6 weeks\n\nAcceptance criteria (postconditions):\n+++\n* In the event details with the matching distribution list, the contact is no longer on the distribution list.\n* A corresponding Incident can be found in the contact history.\n+++",
    "raw_text_2": "###\nAs an Event Manager\nI want that in the view Event Details -> Feedback by importing a CSV file, all contacts with positive feedback are automatically put on all distribution lists of the location belonging to the event, so that all participants with positive feedback are informed in the future about all events at the given location.\n###\nExample: Mr. P. Müller visits the OF Karlsruhe and clicks on the link \"Continue to be informed about events from andrena\" in the thank you email after participation. After importing the feedback, the contact P. Müller should be registered for the distribution list of all events in Karlsruhe (OF, Entwicklertage, XPDays, Workshops) and not only for the OF.\n\nHint:\nPositive feedback means: Clicked=Yes, Deregistered=No.\n----\nAcceptance Criteria:\n+++\n1. all contacts with positive feedback are on all distribution lists that belong to the same location as the event associated with the feedback.\n2. all contacts with positive feedback have the corresponding incidents in their contact history.\n+++",
    "score": 0.1996,
    "us_1": "As an Event Manager I want to be able to edit a contact in the contact details view and unsubscribe him manually from an email distribution list, so that I can process unsubscribe requests via other channels (e.g. phone).",
    "us_2": "As an Event Manager I want that in the view Event Details -> Feedback by importing a CSV file, all contacts with positive feedback are automatically put on all distribution lists of the location belonging to the event, so that all participants with positive feedback are informed in the future about all events at the given location."
}

expected_similarity_result_entry_2 = {
    "ac_1": "* In the event details with the matching distribution list, the contact is no longer on the distribution list.\n* A corresponding Incident can be found in the contact history.",
    "ac_2": "1. all contacts with negative feedback are not subscribed to any distribution list.\n2. all contacts with negative feedback have the corresponding incidents in their history. The consequence for the deletion period is that the contact has only 14 days storage period and the previously given permanent storage period is canceled.",
    "id_1": "COMET-183",
    "id_2": "COMET-225",
    "raw_text_1": "###\nAs an Event Manager\nI want to be able to edit a contact in the contact details view and unsubscribe him manually from an email distribution list,\nso that I can process unsubscribe requests via other channels (e.g. phone).\n###\nThe contact should get an Incident (with manual description of the reason).\nStorage period: 6 weeks\n\nAcceptance criteria (postconditions):\n+++\n* In the event details with the matching distribution list, the contact is no longer on the distribution list.\n* A corresponding Incident can be found in the contact history.\n+++",
    "raw_text_2": "###\nAs an event manager\nI want that in the Event Details -> Feedback view, by importing a CSV file, all contacts with negative feedback are automatically removed from all distribution lists, so that these contacts do not accidentally receive an email from andrena.\n###\n_Example: Mr. P. Müller attends the OF Karlsruhe and clicks on the link \"Do not receive any more information\" in the thank you email after attending. After importing the feedback, the contact P. Müller should be removed from all distribution lists._\n\nNotice:\nPositive feedback means: Clicked=Yes, Unsubscribed=No.\n----\nAcceptance Criteria:\n+++\n1. all contacts with negative feedback are not subscribed to any distribution list.\n2. all contacts with negative feedback have the corresponding incidents in their history. The consequence for the deletion period is that the contact has only 14 days storage period and the previously given permanent storage period is canceled.\n+++",
    "score": 0.3124,
    "us_1": "As an Event Manager I want to be able to edit a contact in the contact details view and unsubscribe him manually from an email distribution list, so that I can process unsubscribe requests via other channels (e.g. phone).",
    "us_2": "As an event manager I want that in the Event Details -> Feedback view, by importing a CSV file, all contacts with negative feedback are automatically removed from all distribution lists, so that these contacts do not accidentally receive an email from andrena."
}

expected_similarity_result_entry_3 = {
    "ac_1": "1. all contacts with positive feedback are on all distribution lists that belong to the same location as the event associated with the feedback.\n2. all contacts with positive feedback have the corresponding incidents in their contact history.",
    "ac_2": "1. all contacts with negative feedback are not subscribed to any distribution list.\n2. all contacts with negative feedback have the corresponding incidents in their history. The consequence for the deletion period is that the contact has only 14 days storage period and the previously given permanent storage period is canceled.",
    "id_1": "COMET-224",
    "id_2": "COMET-225",
    "raw_text_1": "###\nAs an Event Manager\nI want that in the view Event Details -> Feedback by importing a CSV file, all contacts with positive feedback are automatically put on all distribution lists of the location belonging to the event, so that all participants with positive feedback are informed in the future about all events at the given location.\n###\nExample: Mr. P. Müller visits the OF Karlsruhe and clicks on the link \"Continue to be informed about events from andrena\" in the thank you email after participation. After importing the feedback, the contact P. Müller should be registered for the distribution list of all events in Karlsruhe (OF, Entwicklertage, XPDays, Workshops) and not only for the OF.\n\nHint:\nPositive feedback means: Clicked=Yes, Deregistered=No.\n----\nAcceptance Criteria:\n+++\n1. all contacts with positive feedback are on all distribution lists that belong to the same location as the event associated with the feedback.\n2. all contacts with positive feedback have the corresponding incidents in their contact history.\n+++",
    "raw_text_2": "###\nAs an event manager\nI want that in the Event Details -> Feedback view, by importing a CSV file, all contacts with negative feedback are automatically removed from all distribution lists, so that these contacts do not accidentally receive an email from andrena.\n###\n_Example: Mr. P. Müller attends the OF Karlsruhe and clicks on the link \"Do not receive any more information\" in the thank you email after attending. After importing the feedback, the contact P. Müller should be removed from all distribution lists._\n\nNotice:\nPositive feedback means: Clicked=Yes, Unsubscribed=No.\n----\nAcceptance Criteria:\n+++\n1. all contacts with negative feedback are not subscribed to any distribution list.\n2. all contacts with negative feedback have the corresponding incidents in their history. The consequence for the deletion period is that the contact has only 14 days storage period and the previously given permanent storage period is canceled.\n+++",
    "score": 0.5476,
    "us_1": "As an Event Manager I want that in the view Event Details -> Feedback by importing a CSV file, all contacts with positive feedback are automatically put on all distribution lists of the location belonging to the event, so that all participants with positive feedback are informed in the future about all events at the given location.",
    "us_2": "As an event manager I want that in the Event Details -> Feedback view, by importing a CSV file, all contacts with negative feedback are automatically removed from all distribution lists, so that these contacts do not accidentally receive an email from andrena."
}

@pytest.fixture
def usSimilarityVsm():
    logger = logging.getLogger("testLogger")
    mapper = FeedUvlMapper(logger)
    return UserStorySimilarityVsm(mapper, 0.1, False, False)

# test measure_all_pairs_similarity method
@pytest.mark.parametrize("us_dataset,expected_similarity_result_entry_1,expected_similarity_result_entry_2,expected_similarity_result_entry_3", [(us_dataset, expected_similarity_result_entry_1, expected_similarity_result_entry_2, expected_similarity_result_entry_3)])
def test_measure_all_pairs_similarity(usSimilarityVsm: UserStorySimilarityVsm, us_dataset, expected_similarity_result_entry_1, expected_similarity_result_entry_2, expected_similarity_result_entry_3):
    result = usSimilarityVsm.measure_all_pairs_similarity(us_dataset)
    assert len(result) == 3
    tc = TestCase()
    tc.maxDiff = None
    tc.assertDictEqual(result[0], expected_similarity_result_entry_1)
    tc.assertDictEqual(result[1], expected_similarity_result_entry_2)
    tc.assertDictEqual(result[2], expected_similarity_result_entry_3)

def test_measure_all_pairs_similarity_one_us(usSimilarityVsm: UserStorySimilarityVsm):
    result = usSimilarityVsm.measure_all_pairs_similarity(["I do not matter"])
    assert result == []

def test_measure_all_pairs_similarity_no_us(usSimilarityVsm: UserStorySimilarityVsm):
    result = usSimilarityVsm.measure_all_pairs_similarity([])
    assert result == []

# test measure_pairwise_similarity method
@pytest.mark.parametrize("us_dataset,expected_similarity_result_entry_1,expected_similarity_result_entry_2", [(us_dataset, expected_similarity_result_entry_1, expected_similarity_result_entry_2)])
def test_measure_pairwise_similarity(usSimilarityVsm: UserStorySimilarityVsm, us_dataset, expected_similarity_result_entry_1, expected_similarity_result_entry_2):
    result, unexistent_ids = usSimilarityVsm.measure_pairwise_similarity(us_dataset, ["COMET-183"], [])
    assert len(result) == 2
    assert unexistent_ids == 0
    tc = TestCase()
    tc.maxDiff = None
    tc.assertDictEqual(result[0], expected_similarity_result_entry_1)
    tc.assertDictEqual(result[1], expected_similarity_result_entry_2)

def test_measure_pairwise_similarity_one_us(usSimilarityVsm: UserStorySimilarityVsm):
    result, unexistent_ids = usSimilarityVsm.measure_pairwise_similarity(["I do not matter"], ["COMET-183"], [])
    assert result == []
    assert unexistent_ids == 0

def test_measure_pairwise_similarity_no_us(usSimilarityVsm: UserStorySimilarityVsm):
    result, unexistent_ids = usSimilarityVsm.measure_pairwise_similarity([], ["COMET-183"], [])
    assert result == []
    assert unexistent_ids == 0

@pytest.mark.parametrize("us_dataset", [us_dataset])
def test_measure_pairwise_similarity_unexistent_id(usSimilarityVsm: UserStorySimilarityVsm, us_dataset):
    result, unexistent_ids = usSimilarityVsm.measure_pairwise_similarity(us_dataset, ["Unexistent-1"], [])
    assert len(result) == 0
    assert unexistent_ids == 1

@pytest.mark.parametrize("us_dataset", [us_dataset])
def test_measure_pairwise_similarity_unexistent_id_not_extracted(usSimilarityVsm: UserStorySimilarityVsm, us_dataset):
    result, unexistent_ids = usSimilarityVsm.measure_pairwise_similarity(us_dataset, ["Unexistent-1"], ["Unexistent-1"])
    assert len(result) == 0
    assert unexistent_ids == 0
