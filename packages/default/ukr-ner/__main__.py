"""
Simple function that performs Named Entity Recognition for the Ukrainian language.
"""

import time
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained('ukr-models/uk-ner')
model = AutoModelForTokenClassification.from_pretrained('ukr-models/uk-ner')

ner = pipeline('ner', model=model, tokenizer=tokenizer)


def main(args):
    """
    The function.
    """

    content = args.get("content", "Привіт, Петро!")

    if not content:
        return {
            'body': {
                'error': 'content is not filled',
            }
        }

    ts_start = time.time()

    response = ner(content)

    ts_end = time.time()

    return {
        'body': {
            'response': response,
            'took_ms': ts_end - ts_start,
        }
    }
