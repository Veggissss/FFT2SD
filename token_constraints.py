import torch
from transformers import AutoTokenizer, StoppingCriteria, LogitsProcessor
from utils.enums import GenerationState
from config import DEBUG_MODE_ENABLED, REDUCE_NULL_BIAS, STRING_GENERATION_ENABLED


class StopOnToken(StoppingCriteria):
    """
    Stopping criteria to stop generation when a specific token is generated.
    """

    def __init__(self, tokenizer, stop_token):
        super().__init__()
        self.stop_token_id = tokenizer.convert_tokens_to_ids(stop_token)
        self.stopped = None

    def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
        """
        Args:
            input_ids: Tensor of shape [batch_size, sequence_length]
            scores: Tensor of shape [batch_size, vocab_size]

        Returns:
            torch.BoolTensor: Tensor of shape [batch_size] where True indicates
            stopping generation for that sequence
        """
        # Init tensor
        if self.stopped is None:
            self.stopped = torch.zeros(input_ids.shape[0], dtype=torch.bool).to(
                input_ids.device
            )

        # Check if the last token is the stop token
        is_last_token = input_ids[:, -1] == self.stop_token_id
        self.stopped = self.stopped | is_last_token

        if self.stopped.all() and DEBUG_MODE_ENABLED:
            print("Stopping generation for all sequences in the batch.")

        return self.stopped


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

        quote_token_id = tokenizer.convert_tokens_to_ids('"')
        quote2_token_id = tokenizer.convert_tokens_to_ids(' "')
        self.quote_tokens = [quote_token_id]
        # If the quote2 token is <unk> or not
        if quote2_token_id != 0:
            self.quote_tokens.append(quote2_token_id)

        self.bracket_end_token_id = tokenizer.convert_tokens_to_ids("}")

    def __call__(self, input_ids, scores):
        """
        Args:
        input_ids [batch_size, sequence_length]
        scores shape [batch_size, config.vocab_size]
        """
        batch_size = scores.shape[0]

        # Decrease preference for null token
        scores[:, self.null_token_id] -= REDUCE_NULL_BIAS

        for i in range(batch_size):
            if not i in self.state:
                self.state[i] = GenerationState.WAITING

            # Early exit when not enough tokens are generated for constraints
            if input_ids.shape[1] < 5:
                break

            match self.state[i]:
                case GenerationState.WAITING:
                    # See if "value": is generated (There might be separators before value)
                    recent_tokens_ids = input_ids[:, -5:].tolist()
                    last_items = [sublist[-1] for sublist in recent_tokens_ids]

                    if self.colon_token_id in last_items and any(
                        self.value_token_id in batch_list
                        for batch_list in recent_tokens_ids
                    ):
                        self.state[i] = GenerationState.AWAIT_VALUE
                        scores[i] = add_score_mask(scores[i], self.quote_tokens)

                case GenerationState.AWAIT_VALUE:
                    # If the type is string, allow all tokens if enabled, else just the null token
                    if len(self.allowed_token_ids_list[i]) <= 1:
                        if STRING_GENERATION_ENABLED:
                            continue
                        if DEBUG_MODE_ENABLED:
                            print("String generation disabled.")

                    # If the last token is a quote, allow only the restricted value token
                    self.state[i] = GenerationState.AWAITING_QUOTE
                    scores[i] = add_score_mask(
                        scores[i], self.allowed_token_ids_list[i]
                    )
                    log_token_probabilities(
                        self.tokenizer, scores[i], self.allowed_token_ids_list[i]
                    )

                case GenerationState.AWAITING_QUOTE:
                    self.state[i] = GenerationState.AWAITING_END_BRACKET
                    scores[i] = add_score_mask(scores[i], self.quote_tokens)

                case GenerationState.AWAITING_END_BRACKET:
                    self.state[i] = None
                    scores[i] = add_score_mask(scores[i], [self.bracket_end_token_id])
                    # Stopping criteria should stop the next token generation

                case None:
                    print("Stopping criteria was not set.")
                    scores[i] = add_score_mask(scores[i], [self.tokenizer.eos_token_id])
        return scores


def add_score_mask(
    vocab_scores: torch.LongTensor, allowed_ids: list[int] | int
) -> torch.Tensor:
    """
    Apply a mask to the scores based on the allowed token IDs.
    args:
        vocab_scores: Tensor of token scores [vocab_size]
        allowed_ids: List of allowed token IDs or a single token ID
    """
    mask = torch.full_like(vocab_scores, float("-inf"))
    mask[allowed_ids] = 0
    return vocab_scores + mask


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
    null is always allowed and is always added to the allowed tokens list.
    """
    allowed_token_ids = []

    # Add token ID for null token, if data can't be extracted as its not defined in the input text
    null_token_id = tokenizer.convert_tokens_to_ids("null")
    allowed_token_ids.append(null_token_id)

    match token_type:
        case "int":
            for num in range(100):
                # Convert the number to a string and then to token IDs
                token_ids = tokenizer.encode(str(num), add_special_tokens=False)
                # Only add if number maps to a single token to avoid multi-tokens
                if len(token_ids) == 1:
                    allowed_token_ids.append(token_ids[0])
        case "enum":
            if enums is None:
                raise ValueError("Enums must be provided for enum token type.")
            for enum in enums:
                enum_str = str(enum)
                if enum_str == "None":
                    continue
                token_ids = tokenizer.encode(enum_str, add_special_tokens=False)
                # Should be a single token as its added as a seperate token before training
                if len(token_ids) == 1:
                    allowed_token_ids.append(token_ids[0])
                else:
                    print(f"Enum token '{enum_str}' is not a single token!")
        case "boolean":
            for bool_val in ["true", "false"]:
                token_ids = tokenizer.encode(bool_val, add_special_tokens=False)
                if len(token_ids) == 1:
                    allowed_token_ids.append(token_ids[0])
                else:
                    print(f"Boolean token '{bool_val}' is not a single token!")
    return allowed_token_ids
