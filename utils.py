# utils.py
import json
import re
from pathlib import Path
import spacy

_KB_PATH = Path(__file__).parent / "kb" / "local_kb.json"
nlp = None

def get_nlp():
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    return nlp

def extract_entities(text):
    doc = get_nlp()(text)
    return [ent.text for ent in doc.ents]

def load_kb():
    if _KB_PATH.exists():
        with open(_KB_PATH, "r", encoding="utf8") as f:
            return json.load(f)
    return {}

KB = load_kb()

def kb_lookup(entity, target_lang):
    ent = entity.strip()
    if ent in KB and target_lang in KB[ent]:
        return KB[ent][target_lang]
    return None

def apply_kb_on_output(output_text, source_entities, target_lang):
    """
    Replace target mentions with KB canonical forms when possible (case-insensitive).
    """
    adjusted = output_text
    applied = []
    for e in source_entities:
        canonical = kb_lookup(e, target_lang)
        if canonical:
            # naive replacement, preserve word boundaries
            pattern = re.compile(re.escape(e), flags=re.IGNORECASE)
            if pattern.search(adjusted):
                adjusted = pattern.sub(canonical, adjusted)
                applied.append({"entity": e, "canonical": canonical})
    return adjusted, applied
