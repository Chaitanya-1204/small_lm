# To make other scripts importable
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary libraries
import math
import csv
from transformers import  get_scheduler , GPT2Config
import torch
import matplotlib.pyplot as plt

# import custom modules
from utils.all_utils import train , eval , get_dataloaders , get_tokenizer , get_model

#Hyperparameters

seq_length = 256
num_epochs = 20
num_workers = 8
batch_size = 16


# Paths 
train_path = os.path.join("cleaned_datasets" , "train_10M")
tokenizer_path = os.path.join('tokenizer', 'tokenizer_train10M.json')

# model directory and name
model_dir = os.path.join(os.path.dirname(__file__), "models")
model_name = "gpt2-cntx256-param50M-data10M"
last_checkpoint = os.path.join(model_dir, f"{model_name}.pt")
os.makedirs(model_dir, exist_ok=True)

# Logs 
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"{model_name}_training_log.txt")

# Plots 
loss_path = os.path.join(log_dir, f"{model_name}_loss.png")
perplexity_path = os.path.join(log_dir, f"{model_name}_perplexity.png")

# Setting up the device

device =  "cpu" 

# Tokenizer

tokenizer = get_tokenizer(
    tokenizer_path=tokenizer_path,
    seq_length=seq_length
)


# DataLoaders

train_loader , eval_loader , test_loader = get_dataloaders(
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                seq_length=seq_length,
                                                tokenizer=tokenizer,
                                                train_path=train_path,
                                                pad_token_id = tokenizer.pad_token_id
                                            )




# Configure the model parameters
config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions= 2 * tokenizer.model_max_length,
        n_embd= 512,
        n_layer=12,
        n_head= 8,
        n_inner= 2048,
        pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
) 


# Load the model with the specified configuration

model = get_model(config , log_file  , last_checkpoint , device)

total_params = sum(p.numel() for p in model.parameters())

embedding_params = sum(p.numel() for n, p in model.named_parameters() if "wte" in n or "wpe" in n)
non_embedding_params = total_params - embedding_params

# Write initial log information
with open(log_file, "w") as f:
    f.write(f"{'='*60}\n")
    f.write(f"Model Summary Log\n")
    f.write(f"{'='*60}\n")
    f.write(f"Tokenizer Path: {tokenizer_path}\n")
    f.write(f"Model Name: {model_name}\n")
    f.write(f"Sequence Length: {seq_length}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Num Epochs: {num_epochs}\n")
    f.write(f"Warmup Steps: 2000\n")
    f.write(f"Total Training Steps: {len(train_loader) * num_epochs}\n")
    f.write(f"Embedding Parameters: {embedding_params / 1e6:.2f}M\n")
    f.write(f"Non-Embedding Parameters: {non_embedding_params / 1e6:.2f}M\n")
    f.write(f"Total Model Parameters: {total_params / 1e6:.2f}M\n")
    f.write(f"Model Config:\n")
    f.write(f"  - Embedding Dim: {config.n_embd}\n")
    f.write(f"  - Num Layers: {config.n_layer}\n")
    f.write(f"  - Num Heads: {config.n_head}\n")
    f.write(f"  - Inner Dim: {config.n_inner}\n")
    f.write(f"{'='*60}\n\n")