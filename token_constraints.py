import torch
from transformers import AutoTokenizer, StoppingCriteria, LogitsProcessor
from utils.enums import GenerationState
from utils.config import DEBUG_MODE_ENABLED


class TokenTypeConstraintProcessor(LogitsProcessor):
    """
    Logits processor that constrains token generation based on expected value types.
    Controls the sequence: ": "TOKEN"}" where "TOKEN" must be of a specific type.
    """

    def __init__(self, tokenizer: AutoTokenizer, allowed_token_ids: list[int]):
        """
        Initialize the processor with allowed token IDs for a specific type.

        Args:
            tokenizer: Hugging Face tokenizer
            allowed_token_ids: List of token IDs that are allowed for the value
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.allowed_token_ids = allowed_token_ids

        self.quote_token_id = tokenizer.encode('"', add_special_tokens=False)[0]
        self.bracket_end_token_id = tokenizer.encode("}", add_special_tokens=False)[0]
        self.state = GenerationState.WAITING

    def __call__(self, input_ids, scores):
        # Originally for string, only the null token is in allowed_token_ids, so give no constraints
        if len(self.allowed_token_ids) <= 1:
            return scores
        match self.state:
            case GenerationState.WAITING:
                # Decode text up to current point to check context
                text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

                # Check for ': "' indicating that a enum or other constrained token is expected
                if "value" in text and ":" in text and text.rstrip().endswith('"'):
                    # Constrain token values to be filled out
                    scores = add_score_mask(scores, self.allowed_token_ids)
                    self.state = GenerationState.AWAITING_QUOTE
                    # Log probabilities
                    log_token_probabilities(
                        self.tokenizer, scores, self.allowed_token_ids
                    )
            case GenerationState.AWAITING_QUOTE:
                # After the value token, allow only closing quote
                scores = add_score_mask(scores, self.quote_token_id)
                self.state = GenerationState.AWAITING_END_BRACKET

            case GenerationState.AWAITING_END_BRACKET:
                # After closing quote, allow only closing brace
                scores = add_score_mask(scores, self.bracket_end_token_id)
                self.state = None
        return scores


class StopOnToken(StoppingCriteria):
    """
    Stopping criteria to stop generation when a specific token is generated.
    """

    def __init__(self, tokenizer, stop_token):
        super().__init__()
        self.stop_token_id = tokenizer.convert_tokens_to_ids(stop_token)

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1].item() == self.stop_token_id


def add_score_mask(
    scores: torch.LongTensor, allowed_ids: list[int] | int
) -> torch.Tensor:
    """
    Apply a mask to the scores based on the allowed token IDs.
    """
    mask = torch.full_like(scores, float("-inf"))
    mask[:, allowed_ids] = 0
    scores += mask
    return scores


def log_token_probabilities(
    tokenizer: AutoTokenizer,
    scores: torch.Tensor,
    allowed_token_ids: list[int],
    limit: int = 5,
):
    """
    Log the probabilities of the allowed tokens after applying constraints.
    """
    if not DEBUG_MODE_ENABLED:
        return
    # Convert scores to probabilities
    probs = torch.nn.functional.softmax(scores, dim=-1)[0]

    # Get probabilities for allowed tokens and create (token_id, prob) pairs
    token_probs = [(token_id, probs[token_id].item()) for token_id in allowed_token_ids]

    # Sort by probability (descending) and take top 5
    top_tokens = sorted(token_probs, key=lambda x: x[1], reverse=True)[:limit]

    # Print the top 5 tokens with their probabilities
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
                token_id = tokenizer.convert_tokens_to_ids(str(enum))
                allowed_token_ids.append(token_id)
        case "boolean":
            for token_id in range(tokenizer.vocab_size):
                token_str = tokenizer.decode([token_id]).strip()
                if token_str.lower() in ["true", "false"]:
                    allowed_token_ids.append(token_id)
    return allowed_token_ids
