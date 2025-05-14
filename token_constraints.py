import math
import torch
from transformers import AutoTokenizer, StoppingCriteria, LogitsProcessor
from utils.enums import GenerationState
from utils.data_classes import TokenOptions
from config import DEBUG_MODE_ENABLED


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
    Controls the sequence: ": "TOKEN"}" where "TOKEN" must be of a specific type. "TOKEN" might be a series of sub-tokens.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        allowed_token_ids_list: list[list[int]] | list[list[list[int]]],
        token_options: TokenOptions = None,
    ) -> None:
        """
        Initialize the processor with allowed token IDs for a specific type.

        Args:
            tokenizer: Hugging Face tokenizer
            allowed_token_ids_list: List of json entries (the batch) that contains a list of token IDs that are allowed for the value.
            Each token ID might be a list of sub-tokens.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.allowed_token_ids_list = allowed_token_ids_list
        self.state = {}

        # Set token options or default values
        if token_options is not None:
            self.generate_strings = token_options.generate_strings
            self.reduce_null_bias = token_options.reduce_null_bias
        else:
            self.generate_strings = False
            self.reduce_null_bias = 0.0

        # (All values will be forced as strings and converted to the correct type later)
        self.quote_token_id = tokenizer.convert_tokens_to_ids('"')
        self.bracket_end_token_id = tokenizer.convert_tokens_to_ids("}")

        if self.quote_token_id == self.tokenizer.unk_token_id:
            raise ValueError("Quote token ID is not valid!")
        if self.bracket_end_token_id == self.tokenizer.unk_token_id:
            raise ValueError("Bracket end token ID is not valid!")

    def _is_value_pattern_found(self, input_ids: torch.LongTensor) -> bool:
        """Check if the "value": pattern is detected in recent tokens"""
        recent_tokens_ids = input_ids[:, -5:].tolist()
        last_items = [sublist[-1] for sublist in recent_tokens_ids]
        last_items_str = self.tokenizer.decode(last_items, skip_special_tokens=True)

        return ":" in last_items_str and any(
            "value" in self.tokenizer.decode(batch_list, skip_special_tokens=True)
            for batch_list in recent_tokens_ids
        )

    def _is_string_type_enabled(self, batch_index: int) -> bool:
        """
        Handle when token type is a string, giving it unrestricted tokens.
        Will just produce a 'null' value if self.generate_strings is false.
        """
        if len(self.allowed_token_ids_list[batch_index]) <= 1:
            if self.generate_strings:
                return True
            if DEBUG_MODE_ENABLED:
                print("String generation disabled.")
        return False

    def _get_allowed_tokens_for_value(
        self, batch_index: int, last_token_ids: list[int]
    ) -> list[int]:
        """Get the allowed tokens for value generation based on the last_token_id"""
        allowed_token_ids = []
        last_token_decoded = self.tokenizer.decode(
            last_token_ids, skip_special_tokens=False
        )[-1]

        # If the last token is not a quote, check for sub-tokens
        if last_token_decoded != '"':
            if self.quote_token_id in last_token_ids:
                # Reverse the list to find and find the last quote token index
                quote_index = len(last_token_ids) - last_token_ids[::-1].index(
                    self.quote_token_id
                )
                # Only get the generated value tokens without quotes
                last_token_ids = last_token_ids[quote_index:]

            sub_token_ids = self._get_next_subtokens(batch_index, last_token_ids)
            if len(sub_token_ids) > 0:
                # If a complete token has been generated, allow quotes
                for token in self.allowed_token_ids_list[batch_index]:
                    if isinstance(token, list):
                        continue
                    if token in last_token_ids:
                        if self.quote_token_id not in allowed_token_ids:
                            allowed_token_ids.append(self.quote_token_id)
                        self.state[batch_index] = GenerationState.GENERATING_VALUE
                        break

                allowed_token_ids.extend(sub_token_ids)
            else:
                # If a value has been generated and no sub token is found, only allow quotes next and then the end bracket token
                allowed_token_ids = [self.quote_token_id]
                self.state[batch_index] = GenerationState.AWAIT_BRACKET_END
        else:
            # If a complete token is found and a closing quote is generated, allow the end bracket token
            if self.state[batch_index] == GenerationState.GENERATING_VALUE:
                self.state[batch_index] = None
                allowed_token_ids = [self.bracket_end_token_id]
                return allowed_token_ids

            # Allow first tokens in sub-token sequences and all other "full" tokens
            for item in self.allowed_token_ids_list[batch_index]:
                if isinstance(item, list):
                    if item[0] not in allowed_token_ids:
                        allowed_token_ids.append(item[0])
                elif item not in allowed_token_ids:
                    allowed_token_ids.append(item)
        return allowed_token_ids

    def _get_next_subtokens(
        self, batch_index: int, last_value_token_ids: list[int]
    ) -> list[int]:
        """Get the next tokens in multi-token sequences where the last_token_id appears"""
        sub_token_ids = []
        for allowed_token_id in self.allowed_token_ids_list[batch_index]:
            if not isinstance(allowed_token_id, list):
                continue
            # Check if last_value_token_ids is a sublist of allowed_token_id
            for i in range(len(allowed_token_id) - len(last_value_token_ids) + 1):
                if (
                    allowed_token_id[i : i + len(last_value_token_ids)]
                    == last_value_token_ids
                ):
                    # Found matching subsequence, append next token if it exists
                    if i + len(last_value_token_ids) < len(allowed_token_id):
                        next_token = allowed_token_id[i + len(last_value_token_ids)]
                        sub_token_ids.append(next_token)
                        if DEBUG_MODE_ENABLED:
                            print(
                                f"Found matching sequence: {self.tokenizer.decode(last_value_token_ids)} -> {self.tokenizer.decode([next_token])}"
                            )
                    break
        return sub_token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Args:
        input_ids [batch_size, sequence_length]
        logits scores shape [batch_size, config.vocab_size]
        """
        batch_size = scores.shape[0]
        for i in range(batch_size):
            if i not in self.state:
                self.state[i] = GenerationState.WAITING

            # Early exit when not enough tokens are generated for constraints
            if input_ids.shape[1] < 5:
                break

            match self.state[i]:
                case GenerationState.WAITING:
                    # Check if the "value": is generated
                    if not self._is_value_pattern_found(input_ids):
                        continue

                    # Generate first quote token and wait for value
                    self.state[i] = GenerationState.AWAIT_VALUE
                    scores[i] = add_logits_filter_mask(scores[i], [self.quote_token_id])
                case GenerationState.AWAIT_VALUE | GenerationState.GENERATING_VALUE:
                    # Allow unrestricted tokens for string type when enabled
                    if self._is_string_type_enabled(i):
                        continue

                    # NOTE: Some enums might be longer than five tokens, but is still a sub-list of the allowed tokens
                    last_token_ids = input_ids[i, -5:].tolist()
                    allowed_token_ids = self._get_allowed_tokens_for_value(
                        i, last_token_ids
                    )
                    scores[i] = add_logits_filter_mask(scores[i], allowed_token_ids)
                    scores[i] = reduce_null_bias(
                        self.tokenizer, scores[i], threshold=self.reduce_null_bias
                    )

                    # Ignore logging quote token probabilities
                    if allowed_token_ids != [self.quote_token_id]:
                        log_token_probabilities(
                            self.tokenizer, scores[i], allowed_token_ids
                        )

                case GenerationState.AWAIT_BRACKET_END:
                    scores[i] = add_logits_filter_mask(
                        scores[i], [self.bracket_end_token_id]
                    )
                    self.state[i] = None

                case _:
                    # Stopping criteria should stop the generation
                    scores[i] = add_logits_filter_mask(
                        scores[i], [self.tokenizer.unk_token_id]
                    )
        return scores


def add_logits_filter_mask(
    vocab_logits: torch.FloatTensor, allowed_ids: list[int] | int
) -> torch.FloatTensor:
    """
    Apply a filtering mask to the logits based on the allowed token IDs.
    Args:
        vocab_logits: Logits for each token in the vocabulary [vocab_size]
        allowed_ids: List of allowed token IDs
    Returns:
        Masked logits where disallowed tokens are set to -inf.
    """
    mask = torch.full_like(vocab_logits, float("-inf"))
    mask[allowed_ids] = 0
    return vocab_logits + mask


def reduce_null_bias(
    tokenizer: AutoTokenizer, vocab_logits: torch.FloatTensor, threshold: float = 0.8
) -> torch.FloatTensor:
    """Reduce the null token chance if its probability is below a certain threshold."""
    if threshold <= 0.0 or threshold >= 1.0:
        return vocab_logits

    null_token_id = tokenizer.encode("null", add_special_tokens=False)[0]
    probs = torch.nn.functional.softmax(vocab_logits, dim=-1)
    null_probs = probs[null_token_id]

    # Reduce by the set treshold probability
    # If the treshhold is 0.8(80% chance), the null token will be reduced by 80% of its probability
    if null_probs < threshold:
        vocab_logits[null_token_id] -= math.log(1 / (1 - threshold))

    new_probs = torch.nn.functional.softmax(vocab_logits, dim=-1)
    new_null_probs = new_probs[null_token_id]
    print(f"Null token probability: {new_null_probs:.4f} | Original: {null_probs:.4f}")
    return vocab_logits


def log_token_probabilities(
    tokenizer: AutoTokenizer,
    vocab_logits: torch.FloatTensor,
    allowed_token_ids: list[int],
    limit: int = 5,
):
    """
    Log the probabilities of the allowed tokens after applying constraints.
    """
    if not DEBUG_MODE_ENABLED:
        return
    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(vocab_logits, dim=-1)

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
    tokenizer: AutoTokenizer,
    token_type: str,
    enums: list = None,
    include_null: bool = True,
) -> list[int] | list[list[int]]:
    """
    Get the token ids corresponding to the allowed token types.
    null is always allowed and is always added to the allowed tokens list.
    """
    allowed_token_ids = []

    # Add token ID for null token, if data can't be extracted as its not defined in the input text
    if include_null:
        null_token_id = tokenizer.encode("null", add_special_tokens=False)
        if len(null_token_id) == 1:
            allowed_token_ids.append(null_token_id[0])
        else:
            allowed_token_ids.append(null_token_id)
            print(f"Null token {null_token_id} is not a single token!")

    non_single_tokens = []
    match token_type:
        case "int":
            for num in range(1, 100):
                # Convert the number to a string and then to token IDs
                token_ids = tokenizer.encode(str(num), add_special_tokens=False)
                # Only add if number maps to a single token to avoid multi-tokens
                if len(token_ids) == 1:
                    allowed_token_ids.append(token_ids[0])
                else:
                    allowed_token_ids.append(token_ids)
                    non_single_tokens.append(num)
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
                    allowed_token_ids.append(token_ids)
                    non_single_tokens.append(enum_str)
        case "boolean":
            for bool_val in ["true", "false"]:
                token_ids = tokenizer.encode(bool_val, add_special_tokens=False)
                if len(token_ids) == 1:
                    allowed_token_ids.append(token_ids[0])
                else:
                    allowed_token_ids.append(token_ids)
                    non_single_tokens.append(bool_val)
    if DEBUG_MODE_ENABLED and non_single_tokens:
        print(f"Non-single tokens for {token_type}:\n{non_single_tokens}")

    return allowed_token_ids
