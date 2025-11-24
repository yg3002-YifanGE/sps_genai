"""
Assignment 5: GPT-2 Fine-tuned Q&A Model Integration
Provides functions to load and use the fine-tuned GPT-2 model for question answering.
"""

import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional


class GPT2QAModel:
    """
    Wrapper class for fine-tuned GPT-2 model for question answering.
    """
    
    def __init__(self, model_dir: str = "./models/gpt2_squad_finetuned"):
        """
        Initialize the GPT-2 Q&A model.
        
        Args:
            model_dir: Path to the fine-tuned model directory
        """
        self.model_dir = Path(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[GPT2LMHeadModel] = None
        self.tokenizer: Optional[GPT2Tokenizer] = None
        self._is_loaded = False
    
    def load_model(self):
        """
        Load the fine-tuned model and tokenizer.
        Raises RuntimeError if model files are not found.
        """
        if self._is_loaded:
            return
        
        if not self.model_dir.exists():
            raise RuntimeError(
                f"Fine-tuned model not found at {self.model_dir}. "
                f"Please run: python train_gpt2_squad.py first."
            )
        
        print(f"[GPT-2 Q&A] Loading model from {self.model_dir}...")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(str(self.model_dir))
        self.model = GPT2LMHeadModel.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()
        
        self._is_loaded = True
        print(f"[GPT-2 Q&A] Model loaded successfully on {self.device}")
    
    def generate_answer(
        self,
        question: str,
        max_length: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ) -> str:
        """
        Generate an answer to the given question.
        
        Args:
            question: The question to answer
            max_length: Maximum length of generated response
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of responses to generate
        
        Returns:
            Generated answer text
        """
        if not self._is_loaded:
            self.load_model()
        
        # Format the question with our custom prefix
        prompt = f"That is a great question. {question}"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and return the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def generate_multiple_answers(
        self,
        question: str,
        num_responses: int = 3,
        max_length: int = 150,
        temperature: float = 0.8,
    ) -> list[str]:
        """
        Generate multiple diverse answers to the same question.
        
        Args:
            question: The question to answer
            num_responses: Number of different responses to generate
            max_length: Maximum length of each response
            temperature: Sampling temperature (higher = more diverse)
        
        Returns:
            List of generated answer texts
        """
        if not self._is_loaded:
            self.load_model()
        
        prompt = f"That is a great question. {question}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_return_sequences=num_responses,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode all generated sequences
        answers = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return answers


# Global instance (lazy-loaded)
_gpt2_qa_model: Optional[GPT2QAModel] = None


def get_gpt2_qa_model(model_dir: str = "./models/gpt2_squad_finetuned") -> GPT2QAModel:
    """
    Get or create the global GPT-2 Q&A model instance.
    
    Args:
        model_dir: Path to the fine-tuned model directory
    
    Returns:
        GPT2QAModel instance
    """
    global _gpt2_qa_model
    
    if _gpt2_qa_model is None:
        _gpt2_qa_model = GPT2QAModel(model_dir=model_dir)
    
    return _gpt2_qa_model

