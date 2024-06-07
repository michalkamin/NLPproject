<div style="text-align: justify">

# NLP project

Description of the project: 

Fine-tuning T5 and BART models on the task of headline generation. The models are trained on the [BBC news summary dataset](https://www.kaggle.com/datasets/pariza/bbc-news-summary). The dataset contains 2225 articles, along with headline and a summary. The task of fine tuning was done using custom PyTorch lightning classes
- `NewsHeadlineDataset.py` 
- `NewsHeadlineDataModule.py` 
- `HeadlineGenerationModel.py`

The training was done in `project.ipynb`, which is also the main notebook of the project. It contains exemplary dataset analysis, fine-tuning and then experiments. 

Experiments begin with comparing the performance of the fine-tuned models between their pretrained versions. The metrics that we use are BLEU score and ROUGE score. The script which produces the results is `evaluation/evaluate.py` and the function is `calculate_scores_df`. In this version of the experiment, the reference values that are used to calculate the scores are the original headlines and summaries. 

Further experiments were conducted on the [News articles dataset](https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles) - unsseen data for the models. The evaluation was the same as before, but this time there were no summaries in references. 

Furthermore, we decided to use the language_tool_python to calculate the grammar mistakes in the generated headlines.

Final stage of the experiments involved comparison of the fine-tuned models. We decided to incorporate function `evaluation/get_topn_words.py` to force the model to generate headline with most important words from the article. Then the performance was also compared on the BLEU and ROUGE scores. Finally, we decided to check the perplexity of the generated headlines. 


Used datasets:
- https://www.kaggle.com/datasets/pariza/bbc-news-summary
- https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles 

Check the project with flake8:
```bash
flake8 NLPproject
```

Check the project with mypy:
```bash
mypy NLPproject
```

Run the unit tests (inside the project directory):
```bash
python -m unittest discover tests
```
</div>