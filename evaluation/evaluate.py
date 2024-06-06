import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from transformers import (
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    T5Tokenizer,
    BartTokenizer
)
from evaluation.generate_headline import generate_headline
import evaluate

rouge = evaluate.load("rouge")


def calculate_scores_df_tuned(
        df: pd.DataFrame,
        tokenizer_t5: T5Tokenizer,
        trained_t5: T5ForConditionalGeneration,
        tokenizer_bart: BartTokenizer,
        trained_bart: BartForConditionalGeneration,
        include_summary: bool = False,
        verbose: bool = True
) -> pd.DataFrame:
    """
    Calculates BLEU and ROUGE scores for generated headlines in the DataFrame.

    Args:
    df (pd.DataFrame): DataFrame containing articles and original headlines.
        tokenizer_t5: T5 tokenizer.
        trained_t5: T5 model.
        tokenizer_bart: BART tokenizer.
        trained_bart: BART model.
        include_summary (bool): Whether to include summaries for reference.
        verbose (bool): Whether to print progress.

    Returns:
        pd.DataFrame: DataFrame with calculated scores.
    """
    results = []

    for idx, row in df.iterrows():
        if verbose:
            print(f"Processing {idx}/{len(df)}............", end="\r")

        text = row["Article"]
        original_headline = row["Headline"]
        original_headline_tokens = word_tokenize(original_headline.lower())

        if include_summary:
            summary = row["Summary"]
            summary_tokens = word_tokenize(summary.lower())
            references = [original_headline_tokens, summary_tokens]
        else:
            references = [original_headline_tokens]

        finetuned_t5_headline = generate_headline(text, tokenizer_t5, trained_t5)
        finetuned_bart_headline = generate_headline(text, tokenizer_bart, trained_bart)

        finetuned_t5_headline_tokens = word_tokenize(finetuned_t5_headline.lower())
        finetuned_bart_headline_tokens = word_tokenize(finetuned_bart_headline.lower())

        chencherry = SmoothingFunction()

        bleu_finetuned_t5 = sentence_bleu(references, finetuned_t5_headline_tokens, smoothing_function=chencherry.method1)
        bleu_finetuned_bart = sentence_bleu(references, finetuned_bart_headline_tokens, smoothing_function=chencherry.method1)

        scores_finetuned_t5 = rouge.compute(predictions=[finetuned_t5_headline], references=[original_headline])
        rouge_finetuned_t5 = scores_finetuned_t5['rouge1']

        scores_finetuned_bart = rouge.compute(predictions=[finetuned_bart_headline], references=[original_headline])
        rouge_finetuned_bart = scores_finetuned_bart['rouge1']

        results.append({
            'index': idx,
            'original_headline': original_headline,
            'finetuned_t5_headline': finetuned_t5_headline,
            'finetuned_bart_headline': finetuned_bart_headline,
            'bleu_finetuned_t5': bleu_finetuned_t5,
            'bleu_finetuned_bart': bleu_finetuned_bart,
            'rouge_finetuned_t5': rouge_finetuned_t5,
            'rouge_finetuned_bart': rouge_finetuned_bart,
        })

    results_df = pd.DataFrame(results)

    return results_df


def calculate_scores_df(
    df: pd.DataFrame,
    tokenizer_t5: T5Tokenizer,
    pretrained_t5: T5ForConditionalGeneration,
    trained_t5: T5ForConditionalGeneration,
    tokenizer_bart: BartTokenizer,
    pretrained_bart: BartForConditionalGeneration,
    trained_bart: BartForConditionalGeneration,
    include_summary: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Calculate BLEU and ROUGE scores for generated headlines compared to original headlines.

    Args:
        df (pd.DataFrame): DataFrame containing articles and their corresponding original headlines.
        include_summary (bool, optional): Whether to include summary in the references for BLEU calculation. Defaults to False.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing original headlines, generated headlines, and their BLEU and ROUGE scores.
    """
    results = []

    for idx, row in df.iterrows():
        if verbose:
            print(f"Processing {idx}/{len(df)}............", end="\r")

        text = row["Article"]
        original_headline = row["Headline"]
        original_headline_tokens = word_tokenize(original_headline.lower())

        if include_summary:
            summary = row["Summary"]
            summary_tokens = word_tokenize(summary.lower())
            references = [original_headline_tokens, summary_tokens]
        else:
            references = [original_headline_tokens]

        finetuned_t5_headline = generate_headline(text,
                                                  tokenizer_t5,
                                                  trained_t5)
        pretrained_t5_headline = generate_headline(text,
                                                   tokenizer_t5,
                                                   pretrained_t5,
                                                   prompt="summarize: ")

        finetuned_bart_headline = generate_headline(text,
                                                    tokenizer_bart,
                                                    trained_bart)
        pretrained_bart_headline = generate_headline(text,
                                                     tokenizer_bart,
                                                     pretrained_bart,
                                                     prompt="summarize: ")

        finetuned_t5_headline_tokens = word_tokenize(finetuned_t5_headline.lower())
        pretrained_t5_headline_tokens = word_tokenize(pretrained_t5_headline.lower())
        finetuned_bart_headline_tokens = word_tokenize(finetuned_bart_headline.lower())
        pretrained_bart_headline_tokens = word_tokenize(pretrained_bart_headline.lower())

        chencherry = SmoothingFunction()

        bleu_finetuned_t5 = sentence_bleu(references,
                                          finetuned_t5_headline_tokens,
                                          smoothing_function=chencherry.method1)
        bleu_pretrained_t5 = sentence_bleu(references,
                                           pretrained_t5_headline_tokens,
                                           smoothing_function=chencherry.method1)
        bleu_finetuned_bart = sentence_bleu(references,
                                            finetuned_bart_headline_tokens,
                                            smoothing_function=chencherry.method1)
        bleu_pretrained_bart = sentence_bleu(references,
                                             pretrained_bart_headline_tokens,
                                             smoothing_function=chencherry.method1)

        scores_finetuned_t5 = rouge.compute(predictions=[finetuned_t5_headline],
                                            references=[original_headline])
        rouge_finetuned_t5 = scores_finetuned_t5['rouge1']

        scores_pretrained_t5 = rouge.compute(predictions=[pretrained_t5_headline],
                                             references=[original_headline])
        rouge_pretrained_t5 = scores_pretrained_t5['rouge1']

        scores_finetuned_bart = rouge.compute(predictions=[finetuned_bart_headline],
                                              references=[original_headline])
        rouge_finetuned_bart = scores_finetuned_bart['rouge1']

        scores_pretrained_bart = rouge.compute(predictions=[pretrained_bart_headline],
                                               references=[original_headline])
        rouge_pretrained_bart = scores_pretrained_bart['rouge1']

        results.append({
            'index': idx,
            'original_headline': original_headline,
            'finetuned_t5_headline': finetuned_t5_headline,
            'pretrained_t5_headline': pretrained_t5_headline,
            'finetuned_bart_headline': finetuned_bart_headline,
            'pretrained_bart_headline': pretrained_bart_headline,
            'bleu_finetuned_t5': bleu_finetuned_t5,
            'bleu_pretrained_t5': bleu_pretrained_t5,
            'bleu_finetuned_bart': bleu_finetuned_bart,
            'bleu_pretrained_bart': bleu_pretrained_bart,
            'rouge_finetuned_t5': rouge_finetuned_t5,
            'rouge_pretrained_t5': rouge_pretrained_t5,
            'rouge_finetuned_bart': rouge_finetuned_bart,
            'rouge_pretrained_bart': rouge_pretrained_bart
        })

    results_df = pd.DataFrame(results)

    return results_df
