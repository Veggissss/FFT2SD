from transformers import AutoTokenizer, AutoModel
from config import MODELS_DICT


def get_allowed_tokens(tokenizer: AutoTokenizer, token_type: str, enums: list = None):
    """
    Get the token ids corresponding to the allowed token types.
    """
    allowed_token_ids = []

    match token_type:
        case "int":
            # Identify token IDs corresponding to numeric tokens
            for token_id in range(tokenizer.vocab_size):
                token_str = tokenizer.decode([token_id]).strip()
                if token_str.isdigit():
                    allowed_token_ids.append(token_id)
        case "float":
            # Identify token IDs corresponding to float tokens
            for token_id in range(tokenizer.vocab_size):
                token_str = tokenizer.decode([token_id])
                if is_float(token_str):
                    allowed_token_ids.append(token_id)
        case "enum":
            if enums is None:
                raise ValueError("Enums must be provided for enum token type.")
            for enum in enums:
                token_id = tokenizer.convert_tokens_to_ids(enum)
                allowed_token_ids.append(token_id)
        case "boolean":
            for token_id in range(tokenizer.vocab_size):
                token_str = tokenizer.decode([token_id]).strip()
                if token_str.lower() in ["true", "false"]:
                    allowed_token_ids.append(token_id)

    return allowed_token_ids


def is_float(string):
    """
    Check if a string can be converted to a float.
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def register_enum_tokens(tokenizer: AutoTokenizer, enums: list):
    """
    Get every enum value in json fields and add them as new tokens to the tokenizer.
    """
    tokenizer.add_tokens(enums)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODELS_DICT["trained-encoder"])

    register_enum_tokens(tokenizer, ["Hello", "World"])
    allowed_token_ids = get_allowed_tokens(tokenizer, "enum", ["Hello", "World"])

    assert len(allowed_token_ids) > 0
    print(f"Allowed token IDs: {allowed_token_ids}")

    for token_id in allowed_token_ids:
        token_str = tokenizer.decode([token_id])
        print(f"Token ID: {token_id}, Token: '{token_str}'")
