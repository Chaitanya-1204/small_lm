from transformers import GPT2Config, GPT2LMHeadModel
from transformers import GPT2TokenizerFast
from random import sample , seed 
from torch.utils.data import DataLoader , Subset
import torch
import os
from utils.data import BabyLmDataset , collate_fn
from huggingface_hub import login
from utils.all_utils import train , eval

from transformers import get_scheduler
import random
import matplotlib.pyplot as plt
import csv
import math


seq_length = 128
tokenizer_path = os.path.join('tokenizer', 'tokenizer_train10M.json')

# Load the tokenizer
tokenizer = GPT2TokenizerFast(tokenizer_file = tokenizer_path)
print(f"Tokenizer loaded from {tokenizer_path}")

#Set the extra tokens
tokenizer.bos_token = '<s>'
tokenizer.eos_token = '</s>'
tokenizer.pad_token = '<pad>'


train_path = os.path.join("cleaned_datasets" , "train_10M")
train_dataset = BabyLmDataset(train_path , seq_length , tokenizer ,split = "train" ,  random_chunk=True)

# print(f"Training dataset size: {len(train_dataset)}")
# print(f"Training dataset shape: {train_dataset.data.shape}")

eval_path = os.path.join("cleaned_datasets" , "dev")
full_eval_dataset = BabyLmDataset(eval_path , seq_length , tokenizer , split = "dev", offset=0, random_chunk=False) 

# print(f"Evaluation dataset size: {len(eval_dataset)}")
# print(f"Evaluation dataset shape: {eval_dataset.data.shape}"

# random.seed(42)
# eval_indices = random.sample(range(len(full_eval_dataset)), k=16384)
# eval_dataset = Subset(full_eval_dataset, eval_indices)





test_path = os.path.join("cleaned_datasets" , "test")
test_dataset = BabyLmDataset(test_path , seq_length , tokenizer , split = "test", offset=0, random_chunk=False) 

tokenizer.model_max_length = seq_length

pad_token_id = tokenizer.pad_token_id or 0

train_loader = DataLoader(
    train_dataset,
    batch_size= 64 ,
    shuffle=True,
    num_workers = 8 , 
    pin_memory = True,
    collate_fn=lambda batch: collate_fn(batch, pad_token_id=pad_token_id)
)

eval_loader = DataLoader(
    full_eval_dataset , 
    batch_size = 32, 
    shuffle = False ,
    num_workers = 8 , 
    pin_memory = True, 
   
    collate_fn=lambda batch: collate_fn(batch, pad_token_id=pad_token_id)
    
)

test_loader = DataLoader(
    test_dataset , 
    batch_size = 32, 
    shuffle = False , 
    num_workers = 8 , 
    pin_memory = True,
    collate_fn=lambda batch: collate_fn(batch, pad_token_id=pad_token_id)
    
)

model_dir = "train_10M/models"
model_name = "gpt2-custom-model-128-dropout"
last_checkpoint = os.path.join(model_dir, f"{model_name}.pt")
os.makedirs(model_dir, exist_ok=True)

print(last_checkpoint)
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions= 2 * tokenizer.model_max_length,
    n_embd=1536,
    n_layer=24,
    n_head=16,
    n_inner=6144,# 4 * n_embd
    resid_pdrop=0.1, 
    attn_pdrop=0.1,
    
    
    pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    
)    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device {device}")

model = GPT2LMHeadModel(config)

if os.path.exists(last_checkpoint):
    print(f"Loading model weights from {last_checkpoint}")
    model.load_state_dict(torch.load(last_checkpoint))
else:
    print("Initializing new model")

model = model.to(device)
model.gradient_checkpointing_enable()
print(f"Model Parameters: {model.num_parameters() / 1e6:.2f}M")





optimizer = torch.optim.AdamW(model.parameters() , lr = 1e-4 , weight_decay=0.01, fused = True)




num_epochs = 20
scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=2000,
    num_training_steps=len(train_loader) * num_epochs
)





log_file = "experiment_log_cntx128.txt"
with open(log_file, "w") as f:
    f.write("seq_length,epoch,train_loss,eval_loss,train_perplexity,eval_perplexity\n")

training_loss = []
evaluation_loss = []

best_eval_loss = float('inf')
patience = 3
patience_counter = 0

for epoch in range(num_epochs):
    print("--" * 40)
    train_loss = train(model, train_loader, optimizer, device, epoch , scheduler)
    training_loss.append(train_loss) 
    
    
    eval_loss = eval(model, eval_loader, device)
    evaluation_loss.append(eval_loss)
    
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        patience_counter = 0
        torch.save(model.state_dict(), last_checkpoint)
        print(f"Best model saved to {last_checkpoint} at epoch {epoch}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    train_perplexity = math.exp(train_loss)
    eval_perplexity = math.exp(eval_loss)
    
    with open(log_file, "a") as f:
        f.write(f"{seq_length},{epoch+1},{train_loss:.4f},{eval_loss:.4f},{train_perplexity:.4f},{eval_perplexity:.4f} \n")
  

print("--"*40)   

test_loss = eval(model , test_loader , device)
test_ppl = math.exp(test_loss)

print("--" * 40)


with open(log_file, "a") as f:
    f.write(f"Final Test Loss for seq_length={seq_length}: {test_loss:.4f} and Perplexity is {test_ppl}\n")

# Plotting loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), training_loss, label='Training Loss')
plt.plot(range(1, num_epochs + 1), evaluation_loss, label='Evaluation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss')
plt.legend()
plt.grid(True)
plt.savefig("train10M_loss_cntx128.png")
print("Saved loss plot to loss.png")



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
plt.savefig("train10M_perplexity_cntx128.png")
