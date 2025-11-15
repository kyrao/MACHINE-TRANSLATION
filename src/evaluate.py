# src/evaluate.py
from datasets import load_metric
from utils import extract_entities
import sacrebleu

def compute_bleu(hyps, refs):
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu.score

def hallucination_rate(srcs, hyps):
    # HR: fraction of entities in hypotheses not grounded in source or KB
    total_hyp_entities = 0
    hallucinated = 0
    for s, h in zip(srcs, hyps):
        src_ents = set(e.lower() for e in extract_entities(s))
        hyp_ents = set(e.lower() for e in extract_entities(h))
        total_hyp_entities += len(hyp_ents)
        for ent in hyp_ents:
            if ent not in src_ents:
                hallucinated += 1
    if total_hyp_entities == 0:
        return 0.0
    return hallucinated / total_hyp_entities

def factual_consistency_score(srcs, hyps):
    # Simple heuristic: fraction of source entities preserved in hypothesis
    total = 0
    matched = 0
    for s, h in zip(srcs, hyps):
        src_ents = set(e.lower() for e in extract_entities(s))
        hyp_ents = set(e.lower() for e in extract_entities(h))
        if not src_ents:
            continue
        total += len(src_ents)
        matched += sum(1 for e in src_ents if any(e in he for he in hyp_ents))
    if total == 0:
        return 1.0
    return matched / total
