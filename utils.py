# utils.py
import json
import re
from pathlib import Path

_KB_PATH = Path(__file__).parent / "kb" / "local_kb.json"

def extract_entities(text):
    """
    Extract ONLY proper nouns / named entities:
    - Must contain at least one space (multi-word proper names)
    - OR single capitalized words NOT in a stopword list
    """
    stopwords = {
        "I","We","You","He","She","It","They",
        "My","Your","His","Her","Their","Our",
        "A","An","The","In","On","At","To",
        "Want","Eat","Breakfast","Lunch","Dinner"
    }

    words = text.split()
    entities = []

    for i in range(len(words)):
        w = words[i].strip()

        # Ignore all-uppercase words (I, WANT, EAT, etc.)
        if w.upper() == w:
            continue

        # Single capitalized word (India, Paris, Obama)
        if w[0].isupper() and w not in stopwords:
            # Check multi-word name: Barack Obama, New York
            if i + 1 < len(words) and words[i+1][0].isupper():
                entities.append(w + " " + words[i+1])
            else:
                # Only real proper nouns, not small words like "To", "At"
                if len(w) > 3:
                    entities.append(w)

    return list(set(entities))


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
    adjusted = output_text
    applied = []
    for e in source_entities:
        canonical = kb_lookup(e, target_lang)
        if canonical:
            pattern = re.compile(re.escape(e), flags=re.IGNORECASE)
            if pattern.search(adjusted):
                adjusted = pattern.sub(canonical, adjusted)
                applied.append({"entity": e, "canonical": canonical})
    return adjusted, applied
