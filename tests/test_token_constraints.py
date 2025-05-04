import pytest
import torch
from transformers import AutoTokenizer, AddedToken

from token_constraints import (
    get_allowed_tokens,
    TokenTypeConstraintProcessor,
    StopOnToken,
)
from config import MODELS_DICT
from utils.enums import ModelType, GenerationState


def test_allowed_tokens_enum():
    test_enums = ["Hello", "World", "Test", "59jfa9fjFJFj29", "null"]
    test_tokens = [AddedToken(enum, single_word=True) for enum in test_enums]
    tokenizer = AutoTokenizer.from_pretrained(
        MODELS_DICT[ModelType.ENCODER][0].model_name
    )
    tokenizer.add_tokens(test_tokens)
    allowed_token_ids = get_allowed_tokens(tokenizer, "enum", test_enums)

    assert len(allowed_token_ids) > 0, "No allowed token IDs found"

    for token_id in allowed_token_ids:
        token_str = tokenizer.decode([token_id])
        assert token_str in test_enums, f"Unexpected token: '{token_str}'"


def test_allowed_tokens_boolean():
    test_booleans = ["true", "false", "null"]
    tokenizer = AutoTokenizer.from_pretrained(
        MODELS_DICT[ModelType.ENCODER][0].model_name
    )
    tokenizer.add_tokens(test_booleans)
    allowed_token_ids = get_allowed_tokens(tokenizer, "boolean")

    assert len(allowed_token_ids) > 0, "No allowed token IDs found"

    for token_id in allowed_token_ids:
        token_str = tokenizer.decode([token_id]).strip().lower()
        assert token_str in test_booleans, f"Unexpected token: '{token_str}'"


def test_allowed_tokens_int():
    test_ints = ["1", "2", "3", "99", "null"]
    test_invalid_ints = ["100", "1000", "994"]
    tokenizer = AutoTokenizer.from_pretrained(
        MODELS_DICT[ModelType.ENCODER][0].model_name
    )
    tokenizer.add_tokens(test_ints)
    tokenizer.add_tokens(test_invalid_ints)
    allowed_token_ids = get_allowed_tokens(tokenizer, "int")

    assert len(allowed_token_ids) > 0, "No allowed token IDs found"

    allowed_ints = [
        tokenizer.decode(token_id).strip() for token_id in allowed_token_ids
    ]
    for test_int in test_ints:
        # Test out of range integers
        assert test_invalid_ints not in allowed_ints
        assert test_int in allowed_ints, f"Unexpected token: '{test_int}'"


def test_stop_on_token_call():
    tokenizer = AutoTokenizer.from_pretrained(
        MODELS_DICT[ModelType.ENCODER][0].model_name
    )
    stop_token = "}"
    stop_token_id = tokenizer.convert_tokens_to_ids(stop_token)
    stop_criteria = StopOnToken(tokenizer, stop_token)

    # Create test input_ids with 2 sequences
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    scores = torch.randn(2, tokenizer.vocab_size)

    # First call without stop token
    result = stop_criteria(input_ids, scores)
    assert not result.all()
    # Verify shape
    assert stop_criteria.stopped.shape == (2,)

    # Change last token in first sequence to be stop token
    input_ids = torch.tensor([[1, 2, stop_token_id], [4, 5, 6]])
    result = stop_criteria(input_ids, scores)
    assert result[0].item() and not result[1].item()

    # Update last token in second sequence to be stop token
    input_ids = torch.tensor([[1, 2, stop_token_id], [4, 5, stop_token_id]])
    result = stop_criteria(input_ids, scores)
    assert result[0].item() and result[1].item()


def test_token_constraint_per_sequence():
    tokenizer = AutoTokenizer.from_pretrained(
        MODELS_DICT[ModelType.ENCODER_DECODER][0].model_name
    )
    true_token_ids = tokenizer.encode("true", add_special_tokens=False)
    false_token_ids = tokenizer.encode("false", add_special_tokens=False)

    start_tokens = []
    for token in ['"', "value", '"', ":"]:
        token_ids = tokenizer(token, add_special_tokens=False).input_ids
        start_tokens.extend(token_ids)

    # Create allowed token IDs list for 1 sequence
    allowed_token_ids_list = [[true_token_ids, false_token_ids]]
    processor = TokenTypeConstraintProcessor(tokenizer, allowed_token_ids_list)

    # Create two input_ids with "value" and ":" tokens
    input_ids = torch.tensor([start_tokens, start_tokens])

    batch_size = 2
    vocab_size = tokenizer.vocab_size
    scores = torch.randn(batch_size, vocab_size)
    original_scores = scores.clone()

    # Should change to state AWAIT_VALUE
    new_scores = processor(input_ids, scores.clone())
    assert processor.state[0] == GenerationState.AWAIT_VALUE
    assert processor.state[1] == GenerationState.AWAIT_VALUE

    # Check that scores are modified to only allow quote tokens
    for i in range(batch_size):
        for j in range(vocab_size):
            if j == processor.quote_token_id:
                assert new_scores[i][j] == original_scores[i][j]
            else:
                assert new_scores[i][j] == float("-inf") + original_scores[i][j]


def test_token_constraint_full_flow():
    tokenizer = AutoTokenizer.from_pretrained(
        MODELS_DICT[ModelType.DECODER][0].model_name
    )
    true_token_ids = tokenizer.encode("true", add_special_tokens=False)
    false_token_ids = tokenizer.encode("false", add_special_tokens=False)

    start_tokens = []
    for token in ["{", '"', "value", '"', ":"]:
        token_ids = tokenizer(token, add_special_tokens=False).input_ids
        start_tokens.extend(token_ids)

    # Create allowed token IDs list for 1 sequence
    allowed_token_ids_list = [[true_token_ids, false_token_ids]]
    print(f"Allowed token IDs list: {allowed_token_ids_list}")

    processor = TokenTypeConstraintProcessor(tokenizer, allowed_token_ids_list)
    quote_token_id = processor.quote_token_id

    vocab_size = tokenizer.vocab_size
    scores = torch.randn(1, vocab_size)

    # Initial state (WAITING->AWAITING_VALUE)
    input_ids = torch.tensor([start_tokens])
    new_scores = processor(input_ids, scores.clone())
    assert processor.state[0] == GenerationState.AWAIT_VALUE

    # After first quote (AWAITING_VALUE)
    input_ids = torch.tensor([start_tokens + [quote_token_id]])
    new_scores = processor(input_ids, scores.clone())
    assert processor.state[0] == GenerationState.AWAIT_VALUE

    # Check that only allowed tokens have valid scores
    for token_id in range(vocab_size):
        if token_id in [true_token_ids[0], false_token_ids[0]]:
            assert new_scores[0][token_id] > float("-inf")
        else:
            assert new_scores[0][token_id] == float("-inf") + scores[0][token_id]

    # After value token (AWAITING_END_BRACKET)
    input_ids = torch.tensor([start_tokens + [quote_token_id] + true_token_ids])
    new_scores = processor(input_ids, scores.clone())
    assert processor.state[0] == GenerationState.AWAIT_BRACKET_END

    # Check that only quote tokens have valid scores
    for token_id in range(vocab_size):
        if token_id == quote_token_id:
            assert new_scores[0][token_id] > float("-inf")
        else:
            assert new_scores[0][token_id] == float("-inf") + scores[0][token_id]


if __name__ == "__main__":
    pytest.main()
