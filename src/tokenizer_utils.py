# src/tokenizer_utils.py
from transformers import AutoTokenizer
import csv
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Example:
    src: str
    tgt: str
    lang: str

def load_synthetic_parallel(path: str):
    examples = []
    with open(path, encoding='utf8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            examples.append(Example(src=r['src'], tgt=r['tgt'], lang=r.get('lang','')))
    return examples

def prepare_examples_for_model(tokenizer, examples, max_length=128):
    inputs = tokenizer([e.src for e in examples], truncation=True, padding='longest', max_length=max_length)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer([e.tgt for e in examples], truncation=True, padding='longest', max_length=max_length)
    inputs['labels'] = labels['input_ids']
    return inputs
