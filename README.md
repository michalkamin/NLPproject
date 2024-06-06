# NLP project

Description of the project: Fine-tuning T5 and BART models on the task of headline generation. Comparing the performance of the fine-tuned models between their pretrained versions and between each other. Checking the result using various methods like BLEU score or ROUGE score. 

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