# src/train.py
import os
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from src.tokenizer_utils import load_synthetic_parallel, prepare_examples_for_model
from src.contrastive import compute_entity_contrastive_loss

def build_dataset(tokenizer, csv_path):
    examples = load_synthetic_parallel(csv_path)
    enc = prepare_examples_for_model(tokenizer, examples, max_length=128)
    ds = Dataset.from_dict(enc)
    return ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Helsinki-NLP/opus-mt-en-hi")
    parser.add_argument("--train_csv", default="data/parallel_small.csv")
    parser.add_argument("--output_dir", default="models/demo")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--use_eact", action="store_true", help="enable EACT contrastive loss")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    ds = build_dataset(tokenizer, args.train_csv)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        predict_with_generate=False,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # custom compute_loss via callbacks is complicated with HF Trainer; we implement
    # a simple training loop wrapper when EACT is enabled
    if not args.use_eact:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=ds,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        trainer.train()
        trainer.save_model(args.output_dir)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=data_collator)
        model.train()
        for epoch in range(args.num_train_epochs):
            for batch in dataloader:
                # move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                outputs = model(**batch, labels=labels)
                loss_ce = outputs.loss
                # compute EACT contrastive loss on raw text samples (we need original sentences)
                # for demo, pull the raw texts from dataset (not ideal for large scale)
                # We'll approximate using decode of input_ids
                batch_src_texts = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                batch_tgt_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
                loss_entity = compute_entity_contrastive_loss(model, tokenizer, batch_src_texts, batch_tgt_texts, device)
                total_loss = loss_ce + 0.3 * loss_entity
                optim.zero_grad()
                total_loss.backward()
                optim.step()
            print(f"Epoch {epoch+1} done, saving model")
            model.save_pretrained(os.path.join(args.output_dir, f"epoch{epoch+1}"))
        # save final
        model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
