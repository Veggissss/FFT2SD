import torch
from transformers import AutoTokenizer, StoppingCriteria, LogitsProcessor
from utils.enums import GenerationState
from utils.config import DEBUG_MODE_ENABLED, REDUCE_NULL_BIAS


class StopOnToken(StoppingCriteria):
    """
    Stopping criteria to stop generation when a specific token is generated for all sequences in a batch.
    """

    def __init__(self, tokenizer, stop_token):
        super().__init__()
        self.stop_token_id = tokenizer.convert_tokens_to_ids(stop_token)
        self.stopped = None

    def __call__(self, input_ids, scores, **kwargs):
        """
        Args:
        input_ids [batch_size, sequence_length]
        scores [batch_size, vocab_size]
        """
        if self.stopped is None:
            self.stopped = torch.zeros(input_ids.shape[0], dtype=torch.bool).to(
                input_ids.device
            )

        is_last_token = input_ids[:, -1] == self.stop_token_id
        self.stopped = self.stopped | is_last_token
        is_finished = self.stopped.all()

        # Stop generation and reset state
        if is_finished and DEBUG_MODE_ENABLED:
            print("Stopping generation for all sequences in the batch.")

        # Stop generation if the stop token is generated for all sequences in the batch
        return is_finished


class TokenTypeConstraintProcessor(LogitsProcessor):
    """
    Logits processor that constrains token generation based on expected value types.
    Controls the sequence: ": "TOKEN"}" where "TOKEN" must be of a specific type.
    """

    def __init__(
        self, tokenizer: AutoTokenizer, allowed_token_ids_list: list[list[int]]
    ):
        """
        Initialize the processor with allowed token IDs for a specific type.

        Args:
            tokenizer: Hugging Face tokenizer
            allowed_token_ids_list: List of json entries (the batch) that contains a list of token IDs that are allowed for the value
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.allowed_token_ids_list = allowed_token_ids_list
        self.state = {}

        # Identifying sequence of tokens that indicate the value field
        self.value_token_id = tokenizer.convert_tokens_to_ids("value")
        self.null_token_id = tokenizer.convert_tokens_to_ids("null")
        self.colon_token_id = tokenizer.convert_tokens_to_ids(":")
        self.quote_token_id = tokenizer.convert_tokens_to_ids('"')
        self.quote2_token_id = tokenizer.convert_tokens_to_ids(' "')
        self.bracket_end_token_id = tokenizer.convert_tokens_to_ids("}")

    def __call__(self, input_ids, scores):
        """
        Args:
        input_ids [batch_size, sequence_length]
        scores shape [batch_size, config.vocab_size]
        """
        for i, allowed_token_ids in enumerate(self.allowed_token_ids_list):
            if not i in self.state:
                self.state[i] = GenerationState.WAITING

            # Originally for string, only the null token is in allowed_token_ids, so give no constraints
            if len(allowed_token_ids) <= 1 or input_ids.shape[1] < 5:
                continue

            # Decrease preference for null token
            scores[i, self.null_token_id] -= REDUCE_NULL_BIAS

            last_token_id = input_ids[i, -1].item()
            print(f"{i}: {self.state[i]}")
            match self.state[i]:
                case GenerationState.WAITING:
                    # See if "value": is generated (There might be separators before value)
                    recent_tokens = input_ids[i, -5:].tolist()
                    if (
                        last_token_id == self.colon_token_id
                        and self.value_token_id in recent_tokens
                    ):
                        self.state[i] = GenerationState.AWAIT_VALUE
                        scores[i] = add_score_mask(
                            scores[i], [self.quote_token_id, self.quote2_token_id]
                        )

                case GenerationState.AWAIT_VALUE:
                    # If the last token is a quote, allow only the restricted value token
                    self.state[i] = GenerationState.AWAITING_QUOTE
                    scores[i] = add_score_mask(scores[i], allowed_token_ids)
                    log_token_probabilities(
                        self.tokenizer, scores[i], allowed_token_ids
                    )

                case GenerationState.AWAITING_QUOTE:
                    self.state[i] = GenerationState.AWAITING_END_BRACKET
                    scores[i] = add_score_mask(
                        scores[i], [self.quote_token_id, self.quote2_token_id]
                    )

                case GenerationState.AWAITING_END_BRACKET:
                    self.state[i] = GenerationState.AWAITING_EOS
                    scores[i] = add_score_mask(scores[i], self.bracket_end_token_id)

                case GenerationState.AWAITING_EOS:
                    scores[i] = add_score_mask(scores[i], self.tokenizer.eos_token_id)
        return scores


def add_score_mask(
    vocab_scores: torch.LongTensor, allowed_ids: list[int]
) -> torch.Tensor:
    """
    Apply a mask to the scores based on the allowed token IDs.
    args:
        vocab_scores: Tensor of token scores [vocab_size]
        allowed_ids: List of allowed token IDs or a single token ID
    """
    mask = torch.full_like(vocab_scores, float("-inf"))
    mask[allowed_ids] = 0
    vocab_scores += mask
    return vocab_scores


def log_token_probabilities(
    tokenizer: AutoTokenizer,
    vocab_scores: torch.Tensor,
    allowed_token_ids: list[int],
    limit: int = 5,
):
    """
    Log the probabilities of the allowed tokens after applying constraints.
    """
    if not DEBUG_MODE_ENABLED:
        return
    # Convert scores to probabilities
    probs = torch.nn.functional.softmax(vocab_scores, dim=-1)

    # Get probabilities for allowed tokens and create (token_id, prob) pairs
    token_probs = [(token_id, probs[token_id].item()) for token_id in allowed_token_ids]

    # Sort by probability (descending) and take top 5
    top_tokens = sorted(token_probs, key=lambda x: x[1], reverse=True)[:limit]

    # Print the top 5 tokens with their probabilities
    print("===== Top Tokens =====")
    for token_id, prob in top_tokens:
        token_str = tokenizer.decode([token_id])
        print(f"[{token_id}] {prob:.4f} : {token_str}")


def get_allowed_tokens(
    tokenizer: AutoTokenizer, token_type: str, enums: list = None
) -> list[int]:
    """
    Get the token ids corresponding to the allowed token types.
    null is always allowed and added to list.
    """
    allowed_token_ids = []

    # Add token ID for null token, if data can't be extracted as its not defined in the input text
    null_token_id = tokenizer.convert_tokens_to_ids("null")
    allowed_token_ids.append(null_token_id)

    match token_type:
        case "int":
            # Identify token IDs corresponding to numeric tokens
            for token_id in range(tokenizer.vocab_size):
                token_str = tokenizer.decode([token_id]).strip()
                if token_str.isdigit():
                    allowed_token_ids.append(token_id)
        case "enum":
            if enums is None:
                raise ValueError("Enums must be provided for enum token type.")
            for enum in enums:
                enum_str = str(enum)
                if enum_str == "None":
                    continue
                token_id = tokenizer.convert_tokens_to_ids(enum_str)
                allowed_token_ids.append(token_id)
        case "boolean":
            for token_id in range(tokenizer.vocab_size):
                token_str = tokenizer.decode([token_id]).strip()
                if token_str.lower() in ["true", "false"]:
                    allowed_token_ids.append(token_id)
    return allowed_token_ids
