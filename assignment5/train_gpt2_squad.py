"""
Assignment 5: Fine-tuning GPT-2 on SQuAD Dataset (Simplified Version)
不需要 accelerate 包，直接使用 PyTorch 训练循环

Fine-tune openai-community/gpt2 to answer questions in a specific format:
- Start with: "That is a great question."
- End with: "Let me know if you have any other questions."

Training Configuration:
- Default samples: 5,000 (recommended for this assignment)
- Rationale: Sufficient for format learning, computationally practical
- See TRAINING_NOTES.md for detailed justification
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import json
from pathlib import Path


def format_qa_pair(question: str, answer: str) -> str:
    """Format a Q&A pair with the specific format"""
    return f"That is a great question. {question} {answer} Let me know if you have any other questions."


class SQuADDataset(Dataset):
    """PyTorch Dataset for SQuAD Q&A pairs"""
    
    def __init__(self, tokenizer, max_length=512, num_samples=None):
        print("Loading SQuAD dataset...")
        dataset = load_dataset("rajpurkar/squad", split="train")
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            print(f"Using {num_samples} samples")
        
        print(f"Total samples: {len(dataset)}")
        
        # Format and tokenize
        print("Formatting Q&A pairs...")
        self.examples = []
        
        for example in tqdm(dataset):
            question = example["question"]
            answer = example["answers"]["text"][0] if example["answers"]["text"] else ""
            
            if answer:
                text = format_qa_pair(question, answer)
                # Tokenize
                tokens = tokenizer(
                    text + tokenizer.eos_token,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                self.examples.append({
                    'input_ids': tokens['input_ids'].squeeze(),
                    'attention_mask': tokens['attention_mask'].squeeze()
                })
        
        print(f"Prepared {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def train_gpt2_simple(
    output_dir="./models/gpt2_squad_finetuned",
    model_name="openai-community/gpt2",
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    max_length=512,
    num_samples=None,
):
    """
    Fine-tune GPT-2 using simple PyTorch training loop (no Trainer, no accelerate)
    """
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    print(f"\nLoading model: {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.train()
    
    # Prepare dataset
    print("\nPreparing dataset...")
    train_dataset = SQuADDataset(tokenizer, max_length=max_length, num_samples=num_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Total batches per epoch: {len(train_loader)}")
    print("="*70 + "\n")
    
    total_steps = 0
    for epoch in range(num_epochs):
        print(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids  # For language modeling, labels = input_ids
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            total_steps += 1
            
            # Update progress bar
            avg_loss = epoch_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{avg_loss:.4f}'})
            
            # Save checkpoint every 100 steps
            if total_steps % 100 == 0:
                print(f"\n💾 Step {total_steps}: Loss = {loss.item():.4f}, Avg Loss = {avg_loss:.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\n✅ Epoch {epoch + 1} complete - Average Loss: {avg_epoch_loss:.4f}")
    
    # Save final model
    print("\n" + "="*70)
    print("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    training_info = {
        "base_model": model_name,
        "dataset": "rajpurkar/squad",
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "num_samples": num_samples if num_samples else "full dataset",
        "format": "That is a great question. [Q] [A] Let me know if you have any other questions.",
        "final_loss": avg_epoch_loss,
    }
    
    with open(output_path / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"✅ Model saved to: {output_dir}")
    print(f"📊 Training info saved to: {output_path / 'training_info.json'}")
    print("="*70)
    
    return model, tokenizer


def test_model(model_dir="./models/gpt2_squad_finetuned"):
    """Test the fine-tuned model"""
    
    test_questions = [
        "What is machine learning?",
        "Who invented the telephone?",
        "When was Python created?",
        "Where is the Eiffel Tower located?",
        "How does photosynthesis work?",
    ]
    
    print(f"\nLoading model from: {model_dir}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print("\n" + "="*70)
    print("Testing Fine-tuned Model")
    print("="*70)
    
    for question in test_questions:
        prompt = f"That is a great question. {question}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n📝 Question: {question}")
        print(f"💬 Response: {generated_text}")
        print("-"*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on SQuAD (Simple Version)")
    parser.add_argument("--output_dir", type=str, default="./models/gpt2_squad_finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--test_only", action="store_true")
    
    args = parser.parse_args()
    
    if args.test_only:
        test_model(model_dir=args.output_dir)
    else:
        # Train
        model, tokenizer = train_gpt2_simple(
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_length=args.max_length,
            num_samples=args.num_samples,
        )
        
        # Test
        print("\n\n" + "="*70)
        print("Testing the fine-tuned model...")
        print("="*70)
        test_model(model_dir=args.output_dir)

