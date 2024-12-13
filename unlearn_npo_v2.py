import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import math

class JSONLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

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

def compute_npo_loss(model_logits, ref_logits, labels, beta=1.0, temperature=1.0):
    """
    Compute NPO loss based on the paper's formulation
    """
    # Get probability distributions
    model_probs = F.softmax(model_logits / temperature, dim=-1)
    ref_probs = F.softmax(ref_logits / temperature, dim=-1)
    
    # Compute log ratio term
    log_ratio = torch.log(model_probs / (ref_probs + 1e-10) + 1e-10)
    
    # Compute NPO loss according to equation (3) in the paper
    npo_loss = (2/beta) * torch.mean(torch.log(1 + torch.pow(log_ratio, beta)))
    
    return npo_loss

def compute_contrastive_forget_loss(model_outputs, ref_outputs, temperature=0.1):
    """
    Compute contrastive loss to enhance forgetting
    """
    # Normalize embeddings
    model_emb = F.normalize(model_outputs.hidden_states[-1], dim=-1)
    ref_emb = F.normalize(ref_outputs.hidden_states[-1], dim=-1)
    
    # Compute similarity matrix
    similarity = torch.matmul(model_emb, ref_emb.transpose(-2, -1)) / temperature
    
    # Create negative pairs through circular shift
    negative_similarity = torch.roll(similarity, shifts=1, dims=1)
    
    # Compute contrastive loss
    contrastive_loss = -torch.log(
        torch.exp(-similarity.diag()) / 
        (torch.exp(-similarity.diag()) + torch.exp(-negative_similarity.diag()))
    ).mean()
    
    return contrastive_loss

def advanced_npo_unlearning(
    model,
    ref_model,
    tokenizer,
    retain_loader,
    forget_loader,
    output_path,
    lr=1e-4,
    num_steps=50,
    gradient_accumulation_steps=2,
    beta=2.0,
    temperature=2.0,
    retain_weight=1.0,
    forget_weight=1.0,
    contrastive_weight=0.1,
    adaptive_scaling=True,
    target_forget_rate=0.9,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    model.to(device)
    ref_model.to(device)
    ref_model.eval()
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    
    # Initialize adaptive scaling parameters
    if adaptive_scaling:
        scaling_factor = 1.0
        forget_rate_history = []
    
    for step in range(num_steps):
        total_forget_loss = 0.0
        total_retain_loss = 0.0
        
        # Process forget set
        for i, batch in enumerate(forget_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.cuda.amp.autocast():
                # Get outputs from both models
                model_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                with torch.no_grad():
                    ref_outputs = ref_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                
                # Compute NPO loss
                npo_loss = compute_npo_loss(
                    model_outputs.logits,
                    ref_outputs.logits,
                    labels,
                    beta=beta,
                    temperature=temperature
                )
                
                # Compute contrastive loss
                contrastive_loss = compute_contrastive_forget_loss(
                    model_outputs,
                    ref_outputs,
                    temperature=temperature
                )
                
                # Combine losses with adaptive scaling
                if adaptive_scaling:
                    forget_loss = (scaling_factor * npo_loss + 
                                 contrastive_weight * contrastive_loss) * forget_weight
                else:
                    forget_loss = (npo_loss + 
                                 contrastive_weight * contrastive_loss) * forget_weight
            
            if forget_loss is not None:
                scaler.scale(forget_loss).backward()
            
            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_forget_loss += forget_loss.item() if forget_loss is not None else 0
        
        # Process retain set
        for i, batch in enumerate(retain_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                retain_loss = outputs.loss * retain_weight
            
            scaler.scale(retain_loss).backward()
            
            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_retain_loss += retain_loss.item()
        
        # Update adaptive scaling
        if adaptive_scaling and step > 0 and step % 10 == 0:
            current_forget_rate = evaluate_forget_rate(model, forget_loader, device)
            forget_rate_history.append(current_forget_rate)
            
            if current_forget_rate < target_forget_rate:
                scaling_factor *= 1.1  # Increase forgetting pressure
            else:
                scaling_factor *= 0.9  # Decrease forgetting pressure
        
        # Print progress
        avg_forget_loss = total_forget_loss / len(forget_loader)
        avg_retain_loss = total_retain_loss / len(retain_loader)
        print(f"Step {step + 1}/{num_steps} - "
              f"Forget Loss: {avg_forget_loss:.4f}, "
              f"Retain Loss: {avg_retain_loss:.4f}"
              f"{', Scaling: {:.4f}'.format(scaling_factor) if adaptive_scaling else ''}")
        
        torch.cuda.empty_cache()
    
    # Save the final model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Unlearned model saved to {output_path}")

def evaluate_forget_rate(model, forget_loader, device):
    """
    Evaluate the current forgetting rate
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in forget_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.argmax(dim=-1)
            
            # Compare predictions with labels
            correct = (predictions == labels).float().sum()
            total_correct += correct.item()
            total_samples += labels.numel()
    
    model.train()
    return 1 - (total_correct / total_samples)  # Return forget rate
	
	
	
# Example Usage
if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define Hugging Face model and tokenizer
    #model_name = "gpt2"
    #ref_model_name = "gpt2"

    # Load models and tokenizer, keeping initially both the same
    model = AutoModelForCausalLM.from_pretrained('semeval25-unlearning-1B-model')
    ref_model = AutoModelForCausalLM.from_pretrained('semeval25-unlearning-1B-model')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Paths to datasets
    retain_path = "retain_data.jsonl"
    forget_path = "forget_data.jsonl"

    # Create datasets and dataloaders
    retain_dataset = JSONLDataset(retain_path, tokenizer)
    forget_dataset = JSONLDataset(forget_path, tokenizer)

    retain_loader = DataLoader(retain_dataset, batch_size=2, shuffle=True)
    forget_loader = DataLoader(forget_dataset, batch_size=2, shuffle=True)

    # Output path for the unlearned model
    output_model_path = "./unlearned_model"

    # Perform advanced NPO unlearning
    advanced_npo_unlearning(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        output_path=output_model_path,
        lr=1e-4,
        num_steps=50,
        gradient_accumulation_steps=4,
        beta=2.0,
        temperature=1.0,
        retain_weight=1.0,
        forget_weight=1.0,
        contrastive_weight=0.1,
        adaptive_scaling=True,
        target_forget_rate=0.9,
        device=device
    )
