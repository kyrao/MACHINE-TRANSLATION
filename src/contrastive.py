# src/contrastive.py
import torch
import torch.nn.functional as F

def info_nce_loss(anchor, positive, negatives, temperature=0.07):
    """
    anchor: tensor (B, D)
    positive: tensor (B, D)
    negatives: tensor (B, N, D)
    returns: scalar loss
    """
    B, D = anchor.shape
    # normalize
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)
    pos_sim = (anchor * positive).sum(-1) / temperature  # (B,)
    neg_sim = torch.einsum('bd,bnd->bn', anchor, negatives) / temperature  # (B, N)
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+N)
    labels = torch.zeros(B, dtype=torch.long, device=anchor.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def compute_entity_contrastive_loss(model, tokenizer, batch_src_texts, batch_tgt_texts, device, k_neg=4):
    """
    A small heuristic function: compute embeddings for entities by taking mean pooled
    token embeddings for entity spans. For demo, we split by whitespace and take whole-sentence embeddings.
    For production, use exact span alignment and encoder/decoder states.
    """
    # Get sentence embeddings from encoder last hidden-state pooled
    enc = model.get_encoder()
    with torch.no_grad():
        enc_inputs = tokenizer(batch_src_texts, return_tensors='pt', padding=True, truncation=True).to(device)
        enc_outputs = enc(**enc_inputs, return_dict=True)
        src_embeddings = enc_outputs.last_hidden_state.mean(dim=1)  # (B, D)

        dec_inputs = tokenizer(batch_tgt_texts, return_tensors='pt', padding=True, truncation=True).to(device)
        # use encoder outputs for target attention or just run encoder on target (approx)
        dec_out = enc(**dec_inputs, return_dict=True)
        tgt_embeddings = dec_out.last_hidden_state.mean(dim=1)  # (B, D)

    # negatives: shuffle targets to produce k_neg negatives per example
    B = src_embeddings.size(0)
    device = src_embeddings.device
    negatives = []
    for i in range(k_neg):
        idx = torch.randperm(B)
        negatives.append(tgt_embeddings[idx])
    negatives = torch.stack(negatives, dim=1)  # (B, k_neg, D)

    loss = info_nce_loss(src_embeddings, tgt_embeddings, negatives)
    return loss
