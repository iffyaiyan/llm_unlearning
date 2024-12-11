# Right Now I am applying unlearning For Sentence Completion Task:
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import json
import os

class JSONLDataset(Dataset):
    def _init_(self, jsonl_path, tokenizer, max_length=512):
        """
        Dataset for JSONL files.

        Args:
            jsonl_path (str): Path to the JSONL file.
            tokenizer: Tokenizer for encoding the text.
            max_length (int): Maximum sequence length.
        """
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load the JSONL file
        with open(jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                document = item.get("document", "")
                output = item.get("sentence_completion_task", {}).get("output", "")
                if document and output:
                    self.data.append({"input": document, "output": output})

    def _len_(self):
        return len(self.data)

    def _getitem_(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            item["input"], 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length", 
            return_tensors="pt"
        )
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

def gradient_ascent_unlearning(
    model, tokenizer, retain_loader, forget_loader, output_path, lr=1e-4, num_steps=50, device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Perform gradient ascent-based unlearning using the 1B model.

    Args:
        model: Pretrained language model (1B).
        tokenizer: Tokenizer corresponding to the 1B model.
        retain_loader: DataLoader for retain set.
        forget_loader: DataLoader for forget set.
        output_path: Directory to save the updated model.
        lr: Learning rate.
        num_steps: Number of unlearning steps.
        device: Compute device ("cuda" or "cpu").

    Returns:
        None
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(num_steps):
        total_forget_loss = 0.0
        total_retain_loss = 0.0

        # Gradient ascent on the forget set
        for batch in forget_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            forget_loss = outputs.loss

            (-forget_loss).backward()  # Maximize forget loss
            optimizer.step()

            total_forget_loss += forget_loss.item()

        # Gradient descent on the retain set
        for batch in retain_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            retain_loss = outputs.loss

            retain_loss.backward()  # Minimize retain loss
            optimizer.step()

            total_retain_loss += retain_loss.item()

        print(f"Step {step + 1}/{num_steps} - Forget Loss: {total_forget_loss:.4f}, Retain Loss: {total_retain_loss:.4f}")

    # Save the updated model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Unlearned model saved to {output_path}")


# Example Usage
if _name_ == "_main_":
    # Replace <hf_token> with your Hugging Face token
    hf_token = "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"

    # Fetch and load the 1B model and tokenizer
    snapshot_download(repo_id='llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning',
                      token=hf_token, local_dir='semeval25-unlearning-1B-model')
    model = AutoModelForCausalLM.from_pretrained('semeval25-unlearning-1B-model')
    tokenizer = AutoTokenizer.from_pretrained('semeval25-unlearning-1B-model')

    # Paths to datasets
    retain_path = "/teamspace/studios/this_studio/semeval25-unlearning-data/mia_data/member.jsonl"
    forget_path = "/teamspace/studios/this_studio/semeval25-unlearning-data/mia_data/nonmember.jsonl"

    # Create DataLoaders
    retain_dataset = JSONLDataset(retain_path, tokenizer)
    forget_dataset = JSONLDataset(forget_path, tokenizer)
    retain_loader = DataLoader(retain_dataset, batch_size=8, shuffle=True)
    forget_loader = DataLoader(forget_dataset, batch_size=8, shuffle=True)

    # Output path for saving the unlearned model
    output_model_path = "./output/unlearned_1b_model"

    # Run the unlearning process
    gradient_ascent_unlearning(
        model=model,
        tokenizer=tokenizer,
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        output_path=output_model_path,
        lr=1e-4,
        num_steps=50
    )