"""
train.py -- LoRA fine-tuning of Flan-T5-small on complaint summarization.

Task: Given a Barclays customer complaint text, generate a 1-sentence summary.
Model: google/flan-t5-small (77M parameters, manageable on a T4 GPU).
PEFT: LoraConfig with TaskType.SEQ_2_SEQ_LM, targeting q and v projection layers.
Metrics: Inline token-overlap (ROUGE-1 F1 approximation) using numpy only.
         No evaluate library -- incompatible with datasets 4.x.

SageMaker: HuggingFace estimator, ml.g4dn.xlarge (NVIDIA T4 16GB).
           Container: transformers_version=4.56.2, pytorch_version=2.8.0, py_version=py312.
           requirements.txt installs peft>=0.6.0 and numpy<2 into the container.
           Adapter weights saved to /opt/ml/model/ as standard PEFT checkpoint.

eval_strategy="epoch" -- evaluation_strategy removed in transformers 4.41+.
"""

import argparse
import os
import json
import numpy as np

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Flan-T5-small")

    parser.add_argument("--rank",       type=int,   default=8)
    parser.add_argument("--alpha",      type=int,   default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=3e-4)

    parser.add_argument("--max_input_length",  type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=64)

    parser.add_argument("--model-dir",
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output-dir",
                        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"))

    return parser.parse_args()


COMPLAINTS = [
    ("My credit card was charged twice for the same transaction on 12 March and nobody has "
     "refunded me after three calls to your support team over two weeks.",
     "Duplicate charge on 12 March not refunded after three support calls."),
    ("I applied for a graduate current account six weeks ago and still have not received a "
     "decision or any update from the bank despite submitting all required documents.",
     "Graduate account application has had no update after six weeks."),
    ("The mobile app keeps crashing whenever I try to view my savings account balance, "
     "making it impossible to check my money without visiting a branch.",
     "App crashes on savings account balance screen."),
    ("An unauthorised direct debit of 150 pounds was taken from my account on 5 April. "
     "I did not set this up and I need it cancelled and refunded immediately.",
     "Unauthorised 150-pound direct debit taken on 5 April needs cancellation."),
    ("Your overseas transaction fee is 2.99 percent but you charged me 3.5 percent on my "
     "recent trip to Spain. I want an explanation and a refund of the overcharge.",
     "Incorrect overseas fee of 3.5 percent charged instead of 2.99 percent."),
    ("I have been locked out of online banking for five days. The reset password link in "
     "the email you sent does not work and the helpline wait time is over an hour.",
     "Password reset link broken, locked out of online banking for five days."),
    ("My mortgage direct debit failed this month because of a bank error. Your team assured "
     "me it was fixed but I received a late payment notice from the lender.",
     "Bank error caused mortgage direct debit failure and late payment notice."),
    ("I transferred 2000 pounds to my sister's account two days ago and the money has not "
     "arrived. The reference number is not showing in her transaction history.",
     "2000-pound transfer two days ago has not arrived in recipient account."),
    ("A cheque I deposited ten days ago still has not cleared. I need the funds urgently "
     "to pay a supplier invoice that is now overdue.",
     "Cheque deposited ten days ago has not cleared, supplier invoice overdue."),
    ("Your interest rate on my savings account was reduced without any prior notice. "
     "I would have moved the money elsewhere if I had known in advance.",
     "Savings rate reduced without notice, customer would have moved funds."),
    ("I have tried to update my address through online banking four times and it keeps "
     "reverting to my old address. The branch staff said to do it online.",
     "Address update through online banking keeps reverting to old address."),
    ("The contactless limit on my debit card was increased to 100 pounds without my consent. "
     "I want it set back to the original 45-pound limit immediately.",
     "Contactless limit increased to 100 pounds without customer consent."),
    ("My account has been frozen for suspected fraud but all the transactions were genuine. "
     "I cannot pay my rent or buy food while the investigation is ongoing.",
     "Account frozen for fraud investigation blocking rent and food payments."),
    ("I was told on the phone that my loan application was approved but the written "
     "confirmation says declined. I need someone to clarify which decision is correct.",
     "Conflicting loan decision: approved by phone but declined in writing."),
    ("Your ATM at the Oxford Street branch dispensed 50 pounds less than I requested "
     "but debited the full amount from my account.",
     "ATM dispensed 50 pounds short but debited the full requested amount."),
    ("I received a letter saying my account would be closed in 30 days with no reason "
     "given. I have been a customer for 15 years and deserve an explanation.",
     "Account closure notice received with no reason after 15 years as customer."),
    ("The financial hardship support you promised during my call last week has not been "
     "applied to my account and interest is still accruing on my overdraft.",
     "Promised hardship support not applied, overdraft interest still accruing."),
    ("Your chip-and-pin machine at checkout rejected my card three times even though "
     "my account has sufficient funds. I was embarrassed in front of other customers.",
     "Card rejected at chip-and-pin with sufficient funds available."),
    ("I have been billed for a premium banking package I never signed up for. "
     "The charge of 15 pounds per month has been applied for the past six months.",
     "Unsolicited premium package charged at 15 pounds per month for six months."),
    ("The foreign currency I ordered for collection was not ready on the agreed date "
     "and your staff could not tell me when it would arrive.",
     "Ordered foreign currency not ready on agreed collection date."),
]


def augment_pairs(pairs, target=200):
    prefixes = [
        "Complaint: ", "Customer issue: ", "Issue: ", "Problem: ", "Report: ",
        "Concern: ", "Query: ", "", "", "", "",
    ]
    augmented = []
    i = 0
    while len(augmented) < target:
        complaint, summary = pairs[i % len(pairs)]
        prefix = prefixes[i % len(prefixes)]
        augmented.append((prefix + complaint, summary))
        i += 1
    return augmented


class ComplaintDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_input, max_target):
        self.pairs      = pairs
        self.tokenizer  = tokenizer
        self.max_input  = max_input
        self.max_target = max_target

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        complaint, summary = self.pairs[idx]
        input_text  = "summarize: " + complaint
        target_text = summary

        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_input,
            truncation=True,
            padding=False,
        )
        labels = self.tokenizer(
            target_text,
            max_length=self.max_target,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def token_overlap_f1(pred_ids, label_ids, tokenizer):
    """Approximate ROUGE-1 F1 using token-level unigram overlap."""
    f1_scores = []
    for pred, label in zip(pred_ids, label_ids):
        label = [t for t in label if t != -100]
        pred  = [t for t in pred  if t not in (tokenizer.pad_token_id,
                                                tokenizer.eos_token_id)]
        if len(label) == 0 and len(pred) == 0:
            f1_scores.append(1.0)
            continue
        if len(label) == 0 or len(pred) == 0:
            f1_scores.append(0.0)
            continue
        pred_set  = set(pred)
        label_set = set(label)
        overlap   = len(pred_set & label_set)
        precision = overlap / len(pred_set)
        recall    = overlap / len(label_set)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1_scores))


def compute_metrics_factory(tokenizer, max_target):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)
        f1 = token_overlap_f1(preds, labels, tokenizer)
        return {"rouge1_approx": round(f1, 4)}
    return compute_metrics


def main():
    args = parse_args()

    print(f"LoRA rank={args.rank}, alpha={args.alpha}, dropout={args.lora_dropout}")
    print(f"Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Model dir: {args.model_dir}")

    model_name = "google/flan-t5-small"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    total_base = sum(p.numel() for p in base_model.parameters())
    print(f"Base model parameters: {total_base:,}")

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["q", "v"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(base_model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters with LoRA: {trainable:,}")
    print(f"Trainable fraction: {100 * trainable / total_base:.4f}%")

    all_pairs = augment_pairs(COMPLAINTS, target=200)
    split     = int(0.85 * len(all_pairs))
    train_pairs = all_pairs[:split]
    val_pairs   = all_pairs[split:]

    train_dataset = ComplaintDataset(
        train_pairs, tokenizer, args.max_input_length, args.max_target_length)
    val_dataset   = ComplaintDataset(
        val_pairs,   tokenizer, args.max_input_length, args.max_target_length)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rouge1_approx",
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        report_to="none",
        dataloader_pin_memory=False,
    )

    compute_metrics = compute_metrics_factory(tokenizer, args.max_target_length)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting LoRA fine-tuning ...")
    trainer.train()

    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    final_metrics = trainer.evaluate()
    summary = {
        "rank":             args.rank,
        "alpha":            args.alpha,
        "trainable_params": trainable,
        "total_params":     total_base,
        "trainable_pct":    round(100 * trainable / total_base, 4),
        "final_rouge1":     final_metrics.get("eval_rouge1_approx", None),
    }
    with open(os.path.join(args.model_dir, "lora_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Adapter weights saved to {args.model_dir}")
    print(f"Training summary: {summary}")


if __name__ == "__main__":
    main()
