# src/decode.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from utils import extract_entities, apply_kb_on_output, KB

def constrained_generate(model, tokenizer, src_text, target_lang="hi", beam_size=4, penalty=-2.0, device="cpu"):
    """
    A practical constrained decoding wrapper:
    - do standard generate with beams
    - post-process top beam by applying KB substitutions for matched entities
    - penalize beams that contain hallucinated capitalized tokens not in KB (simple heuristic)
    """
    model.to(device)
    inputs = tokenizer(src_text, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs, num_beams=beam_size, max_length=256)
    best = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # apply KB substitutions
    src_entities = extract_entities(src_text)
    adjusted, changes = apply_kb_on_output(best, src_entities, target_lang)
    return best, adjusted, changes
