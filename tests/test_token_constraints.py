import pytest
from transformers import AutoTokenizer, AddedToken
from utils.token_constraints import get_allowed_tokens
from utils.config import MODELS_DICT


def test_allowed_tokens_enum():
    test_enums = ["Hello", "World", "Test", "59jfa9fjFJFj29", "null"]
    test_tokens = [AddedToken(enum, single_word=True) for enum in test_enums]
    tokenizer = AutoTokenizer.from_pretrained(MODELS_DICT["encoder"])
    tokenizer.add_tokens(test_tokens)
    allowed_token_ids = get_allowed_tokens(tokenizer, "enum", test_enums)

    assert len(allowed_token_ids) > 0, "No allowed token IDs found"

    for token_id in allowed_token_ids:
        token_str = tokenizer.decode([token_id])
        assert token_str in test_enums, f"Unexpected token: '{token_str}'"


def test_allowed_tokens_boolean():
    test_booleans = ["true", "false", "null"]
    tokenizer = AutoTokenizer.from_pretrained(MODELS_DICT["encoder"])
    tokenizer.add_tokens(test_booleans)
    allowed_token_ids = get_allowed_tokens(tokenizer, "boolean")

    assert len(allowed_token_ids) > 0, "No allowed token IDs found"

    for token_id in allowed_token_ids:
        token_str = tokenizer.decode([token_id]).strip().lower()
        assert token_str in test_booleans, f"Unexpected token: '{token_str}'"


def test_allowed_tokens_int():
    test_ints = ["1", "2", "3", "100", "1000", "null"]
    tokenizer = AutoTokenizer.from_pretrained(MODELS_DICT["encoder"])
    tokenizer.add_tokens(test_ints)
    allowed_token_ids = get_allowed_tokens(tokenizer, "int")

    assert len(allowed_token_ids) > 0, "No allowed token IDs found"

    allowed_ints = [
        tokenizer.decode([token_id]).strip() for token_id in allowed_token_ids
    ]
    for test_int in test_ints:
        assert test_int in allowed_ints, f"Unexpected token: '{test_int}'"


if __name__ == "__main__":
    pytest.main()
