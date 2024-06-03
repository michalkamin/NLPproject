import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from typing import List, Dict
from nltk.tokenize import word_tokenize
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration, T5Tokenizer, BartTokenizer
from generate_headline import generate_headline

def calculate_scores_df_tuned(df: pd.DataFrame, tokenizer_t5: T5Tokenizer, trained_t5: T5ForConditionalGeneration, tokenizer_bart: BartTokenizer, trained_bart: BartForConditionalGeneration, include_summary: bool = False, verbose: bool = True) -> pd.DataFrame:
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
    rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    for idx, row in df.iterrows():
        if verbose:
            print(f"Processing {idx + 1}/{len(df)}............")

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

        rouge_finetuned_t5 = rouge.score(finetuned_t5_headline, original_headline)['rouge1'].fmeasure
        rouge_finetuned_bart = rouge.score(finetuned_bart_headline, original_headline)['rouge1'].fmeasure

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
