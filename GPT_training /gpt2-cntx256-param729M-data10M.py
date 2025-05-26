# To make other scripts importable
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary libraries
import math
import csv
from transformers import  get_scheduler
import torch
import matplotlib.pyplot as plt

# import custom modules
from utils.all_utils import train , eval , get_dataloaders , get_tokenizer , get_model

#Hyperparameters

seq_length = 256
num_epochs = 20
num_workers = 8
batch_size = 16

print(os.getcwd())
# Paths 
train_path = os.path.join("cleaned_datasets" , "train_10M")
tokenizer_path = os.path.join('tokenizer', 'tokenizer_train10M.json')

# model directory and name
model_dir = os.path.join(os.path.dirname(__file__), "models")
model_name = "gpt2-cntx256-param729M-data10M"
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

device = "cuda:1" if torch.cuda.is_available() else "cpu" 

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



# Get the model and its configuration

model , config = get_model(tokenizer , log_file  , last_checkpoint , device)

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
    f.write(f"Model Config:\n")
    f.write(f"  - Embedding Dim: {config.n_embd}\n")
    f.write(f"  - Num Layers: {config.n_layer}\n")
    f.write(f"  - Num Heads: {config.n_head}\n")
    f.write(f"  - Inner Dim: {config.n_inner}\n")
    f.write(f"{'='*60}\n\n")

# Initialize the optimizer
optimizer = torch.optim.AdamW(model.parameters() , lr =1e-4 , weight_decay = 0.01 , fused = True)

# Initialize the learning rate scheduler
scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=2000,
    num_training_steps=len(train_loader) * num_epochs
)

# Initializing lists to store training and evaluation losses
training_loss = []
evaluation_loss = []

# Initializing variables for early stopping
best_eval_loss = float('inf')
patience = 3
patience_counter = 0

# Start training and evaluation loop
for epoch in range(num_epochs):
    with open(log_file, "a") as f:
        f.write("--" * 40 + "\n")
    
    # Train the model for one epoch
    train_loss = train(model, train_loader, optimizer, device, epoch, scheduler)
    training_loss.append(train_loss) 

    # Evaluate the model on the evaluation set
    eval_loss = eval(model, eval_loader, device)
    evaluation_loss.append(eval_loss)

    # Log the losses and perplexities
    train_perplexity = math.exp(train_loss)
    eval_perplexity = math.exp(eval_loss)

    with open(log_file, "a") as f:
        
        f.write(f"{'='*60}\n")
        
        f.write(f"Training and Evaluation Results for Epoch {epoch+1}\n")
        f.write(f"Seq Length: {seq_length}\n")
        f.write(f"Train Loss: {train_loss:.4f} | Perplexity: {train_perplexity:.4f}\n")
        f.write(f"Eval  Loss: {eval_loss:.4f} | Perplexity: {eval_perplexity:.4f}\n")
        
        f.write(f"{'='*60}\n\n")
        
    # Check if the evaluation loss improved
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        patience_counter = 0
        torch.save(model.state_dict(), last_checkpoint)

           
    else:
        patience_counter += 1
        if patience_counter >= patience:
            with open(log_file, "a") as f:
                f.write(f"Early stopping triggered at epoch {epoch+1}\n")
            break


# Final evaluation on the test set
test_loss = eval(model , test_loader , device)
test_ppl = math.exp(test_loss)

# Log final test results

with open(log_file, "a") as f:
    
    f.write(f"{'='*60}\n")
    
    f.write(f"Final Test Results\n")
    f.write(f"Seq Length: {seq_length}\n")
    f.write(f"Test Loss: {test_loss:.4f} | Perplexity: {test_ppl:.4f}\n")
    
    f.write(f"{'='*60}\n")
    
# Plotting loss curves

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), training_loss, label='Training Loss')
plt.plot(range(1, num_epochs + 1), evaluation_loss, label='Evaluation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss')
plt.legend()
plt.grid(True)
plt.savefig(loss_path)


# plotting perplexity

train_ppl_curve = [math.exp(l) for l in training_loss]
eval_ppl_curve = [math.exp(l) for l in evaluation_loss]

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_ppl_curve) + 1), train_ppl_curve, label='Train Perplexity')
plt.plot(range(1, len(eval_ppl_curve) + 1), eval_ppl_curve, label='Eval Perplexity')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Training and Evaluation Perplexity')
plt.legend()
plt.grid(True)
plt.savefig(perplexity_path)

