from typing import List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_headline(
    text: str,
    tokenizer: PreTrainedTokenizer,
    input_model: PreTrainedModel,
    prompt: str = "",
    min_length: int = 7,
    num_beams: int = 5,
    length_penalty: float = 1.0,
    max_new_tokens: int = 20,
    force_words: Optional[List[str]] = None,
) -> str:
    """
    Generates a headline for the given text using a specified model and tokenizer.
    Optionally enforces the inclusion of specified words in the generated headline.

    Args:
        text (str): The input text for which the headline is to be generated.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        input_model (PreTrainedModel): The pre-trained model used for generation.
        prompt (str, optional): A prompt to prepend to the text. Defaults to ''.
        min_length (int, optional): Minimum length of the generated headline. Defaults to 7.
        num_beams (int, optional): Number of beams for beam search. Defaults to 5.
        length_penalty (float, optional): Length penalty for beam search. Defaults to 1.0.
        max_new_tokens (int, optional): Maximum number of new tokens to generate.
        Defaults to 20.
        force_words (Optional[List[str]], optional): A list of words that must
        be included in the generated headline.
        Defaults to None.

    Returns:
        str: The generated headline as a string.
    """

    input_model.to(device)

    text_encoding = tokenizer(
        prompt + text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    input_ids = text_encoding["input_ids"].to(device)
    attention_mask = text_encoding["attention_mask"].to(device)

    if force_words:
        force_words = [str(word) for word in force_words]
        force_words_ids = [
            tokenizer(word, add_special_tokens=False)["input_ids"]
            for word in force_words
        ]
    else:
        force_words_ids = None

    generated_ids = input_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=True,
        length_penalty=length_penalty,
        force_words_ids=force_words_ids,
    )

    preds = [
        tokenizer.decode(
            gen_id,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        for gen_id in generated_ids
    ]

    return "".join(preds)
