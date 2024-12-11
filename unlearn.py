import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

class JSONLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load the data from the JSONL file
        with open(jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                document = item.get("document", "")
                output = item.get("sentence_completion_task", {}).get("output", "")
                self.data.append({"input": document, "output": output})

        print(f"Loaded {len(self.data)} items.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize the input
        inputs = self.tokenizer(
            item["input"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        # Tokenize the output (labels)
        labels = self.tokenizer(
            item["output"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0)
        }


def kl_divergence(logits, target_probs):
    """
    Calculate KL divergence between the model's logits and target probabilities.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    kl_div = F.kl_div(log_probs, target_probs, reduction="batchmean")
    return kl_div

def gradient_ascent_unlearning_with_kl(
    model, tokenizer, retain_loader, forget_loader, output_path, lr=1e-4, num_steps=50, gradient_accumulation_steps=2, device="cuda" if torch.cuda.is_available() else "cpu"
):
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    for step in range(num_steps):
        total_forget_loss = 0.0
        total_retain_loss = 0.0

        # Gradient ascent on the forget set
        for i, batch in enumerate(forget_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Target distribution: uniform for forgetting
                target_probs = torch.ones_like(logits) / logits.size(-1)
                forget_loss = kl_divergence(logits, target_probs)

            forget_loss = forget_loss / gradient_accumulation_steps
            scaler.scale(-forget_loss).backward()  # Maximize forget loss

            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(forget_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_forget_loss += forget_loss.item()

        # Gradient descent on the retain set
        for i, batch in enumerate(retain_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits

                # Target distribution: labels for retaining
                target_probs = F.one_hot(labels, num_classes=logits.size(-1)).float()
                target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)  # Normalize
                retain_loss = kl_divergence(logits, target_probs)

            retain_loss = retain_loss / gradient_accumulation_steps
            scaler.scale(retain_loss).backward()  # Minimize retain loss

            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(retain_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_retain_loss += retain_loss.item()

        print(f"Step {step + 1}/{num_steps} - Forget Loss: {total_forget_loss:.4f}, Retain Loss: {total_retain_loss:.4f}")

        # Clear cache to free up memory
        torch.cuda.empty_cache()

    # Save the updated model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Unlearned model saved to {output_path}")


# Example Usage
if _name_ == "_main_":
    # Example Usage
    hf_token = "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"

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained('semeval25-unlearning-1B-model')
    tokenizer = AutoTokenizer.from_pretrained('allenai/OLMo-1B-0724-hf')

    # Define paths to the JSONL datasets
    retain_path = "/teamspace/studios/this_studio/unlearning/semeval25-unlearning-data/mia_data/member.jsonl"
    forget_path = "/teamspace/studios/this_studio/unlearning/semeval25-unlearning-data/mia_data/nonmember.jsonl"

    # Initialize datasets and DataLoaders
    retain_dataset = JSONLDataset(retain_path, tokenizer)
    forget_dataset = JSONLDataset(forget_path, tokenizer)
    retain_loader = DataLoader(retain_dataset, batch_size=1, shuffle=True)
    forget_loader = DataLoader(forget_dataset, batch_size=1, shuffle=True)

    # Define output path for the updated model
    output_model_path = "./unlearning/output/unlearned_1b_model"

    # Perform gradient ascent unlearning
    gradient_ascent_unlearning(
        model=model,
        tokenizer=tokenizer,
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        output_path=output_model_path,
        lr=1e-4,
        num_steps=50,
        gradient_accumulation_steps=2
    )