import pytest
from src.techniques.preprocessing import (get_tokenized_list, pos_tagger,
    remove_punctuation, remove_stopwords, retrieve_corpus, word_stemmer,
    remove_us_skeleton, get_us_action)


def test_get_tokenized_list():
    doc_text = "This text should be tokenized. And this too"
    expected_tokenized_list = ["This", "text", "should", "be", "tokenized", ".", "And", "this", "too"]
    result = get_tokenized_list(doc_text)
    assert result == expected_tokenized_list

def test_get_tokenized_list_special_chars():
    doc_text = "comma, . ! ? question? exclamation! colon: 80%"
    expected_tokenized_list = ["comma", ",", ".", "!", "?", "question", "?", "exclamation", "!", "colon", ":", "80", "%"]
    result = get_tokenized_list(doc_text)
    assert result == expected_tokenized_list

def test_get_tokenized_list_from_no_string():
    doc_text = 123
    with pytest.raises(Exception) as e_info:
        get_tokenized_list(doc_text)

def test_word_stemmer():
    tokenized_list = ["Perform", "Stemming", "On", "A", "Tokenized", "List", "."]
    result = word_stemmer(tokenized_list)
    expected_stemmed_list = ["perform", "stem", "on", "a", "token", "list", "."]
    assert result == expected_stemmed_list

def test_word_stemmer_string_input():
    input = "This is only a string."
    with pytest.raises(TypeError) as e_info:
        word_stemmer(input)

def test_pos_tagger():
    tokenized_list = ["The", "boy", "eats", "his", "small", "lunch", "proudly", "."]
    result = pos_tagger(tokenized_list)
    expected_tagged_list = [("The", "DT"), ("boy", "NN"), ("eats", "VBZ"), ("his", "PRP$"), ("small", "JJ"), ("lunch", "NN"), ("proudly", "RB"), (".", ".")]
    assert result == expected_tagged_list

def test_pos_tagger_string_input():
    input = "This is only a string."
    with pytest.raises(TypeError) as e_info:
        pos_tagger(input)

def test_remove_stopwords():
    tokenized_list = ["The", "boy", "eats", "his", "small", "lunch", "proudly", "."]
    result = remove_stopwords(tokenized_list)
    expected_token_list = ["boy", "eats", "small", "lunch", "proudly", "."]
    assert result == expected_token_list

def test_remove_stopwords_string_input():
    input = "This is only a string."
    with pytest.raises(TypeError) as e_info:
        remove_stopwords(input)
    
def test_remove_punctuation():
    doc_text = r"""Remove, al:l punctuation!, also all of this:!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    result = remove_punctuation(doc_text)
    expected_doc_text = "Remove all punctuation also all of this"
    assert result == expected_doc_text

def test_retrieve_corpus():
    us_dataset = [
        {
            "id": "COMET-1",
            "text": "us1",
            "acceptance_criteria": "example acceptance criteria",
            "raw_text": "example raw text"
        },
        {
            "id": "COMET-2",
            "text": "us2",
            "acceptance_criteria": "example acceptance criteria",
            "raw_text": "example raw text"
        },
        {
            "id": "COMET-3",
            "text": "us3",
            "acceptance_criteria": "example acceptance criteria",
            "raw_text": "example raw text"
        }
    ]
    result = retrieve_corpus(us_dataset)
    expected_corpus = ["us1", "us2", "us3"]
    assert result == expected_corpus

def test_retrieve_corpus_missing_field():
    us_dataset = [
        {
            "id": "COMET-1",
            "acceptance_criteria": "example acceptance criteria",
            "raw_text": "example raw text"
        }
    ]
    with pytest.raises(Exception) as e_info:
        retrieve_corpus(us_dataset)

def test_retrieve_corpus_emtpy_list():
    result = retrieve_corpus([])
    assert result == []

# test _get_us_action
def test_get_us_action():
    us = "As a user I wAnt to do that action sO thAT I can achieve this purpose."
    expected_us = "to do that action"
    result = get_us_action(us)
    assert result == expected_us

def test_get_us_action_without_reason():
    us = "As a user I want to do that action."
    expected_us = "to do that action."
    result = get_us_action(us)
    assert result == expected_us

def test_get_us_action_special_chars():
    us = "As a user,\n I want to do that action,\n so that I can achieve this purpose."
    expected_us = "to do that action,"
    result = get_us_action(us)
    assert result == expected_us

def test_get_us_action_error():
    us = "As a user,\n want to do that action,\n so I can achieve this purpose."
    result = get_us_action(us)
    assert result == us

# test _remove_us_skeleton
def test_remove_us_skeleton():
    us = "As a user,\n I want to do that action,\n so that I can achieve this purpose."
    expected_us = "user,\n to do that action,\n I can achieve this purpose."
    result = remove_us_skeleton(us)
    assert result == expected_us

def test_remove_us_skeleton_without_action():
    us = "As a user,\n so that I can achieve this purpose."
    expected_us = "user,\n I can achieve this purpose."
    result = remove_us_skeleton(us)
    assert result == expected_us

def test_remove_us_skeleton_without_skeleton():
    us = "Blu Bli Bli Bla"
    expected_us = "Blu Bli Bli Bla"
    result = remove_us_skeleton(us)
    assert result == expected_us
