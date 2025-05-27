# To make other scripts importable
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import  get_scheduler , GPT2Config , GPT2LMHeadModel
import torch
import matplotlib.pyplot as plt

import math

from utils.all_utils import distill_train , eval , get_dataloaders , get_tokenizer
#Hyperparameters

seq_length = 256
num_epochs = 20
num_workers = 8
batch_size = 16


# Paths 
train_path = os.path.join("cleaned_datasets" , "train_10M")
tokenizer_path = os.path.join('tokenizer', 'tokenizer_train10M.json')


model_dir = os.path.join(os.path.dirname(__file__), "models")
model_name = "distillation_gpt2-param729M-54M-data10M_epochs20"
last_checkpoint = os.path.join(model_dir, f"{model_name}.pt")
os.makedirs(model_dir, exist_ok=True)

# Logs 
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"{model_name}_training_log.txt")



# Setting up the device

device = "cuda" if torch.cuda.is_available() else "cpu" 


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


teacher_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions= 2 * tokenizer.model_max_length,
        n_embd=1536,
        n_layer=24,
        n_head=16,
        n_inner=6144,
        pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
) 

student_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions= 2 * tokenizer.model_max_length,
        n_embd= 512,
        n_layer=12,
        n_head= 8,
        n_inner= 2048,
        pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
) 

teacher = GPT2LMHeadModel(teacher_config)
teacher.load_state_dict(torch.load("GPT_training/models/gpt2-cntx256-param729M-data10M.pt"))
teacher.eval()
teacher.to(device)

# Load student (e.g. smaller model)
student = GPT2LMHeadModel(student_config)
student.load_state_dict(torch.load("GPT_training/models/gpt2-cntx256-param54M-data10M.pt"))
student.to(device)

teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
student_params = sum(p.numel() for p in student.parameters()) / 1e6


# Initialize the optimizer
optimizer = torch.optim.AdamW(student.parameters() , lr =1e-4 , weight_decay = 0.01 , fused = True)

# Initialize the learning rate scheduler
scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=1400,
    num_training_steps=len(train_loader) * num_epochs
)

# Initializing lists to store training and evaluation losses
training_loss = []
evaluation_loss = []

best_eval_loss = float('inf')

# Write initial log information for distillation training
with open(log_file, "w") as f:
    f.write(f"{'='*60}\n")
    f.write(f"Distillation Training Log\n")
    f.write(f"{'='*60}\n")
    f.write(f"Model Name: {model_name}\n")
    f.write(f"Sequence Length: {seq_length}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Num Epochs: {num_epochs}\n")
    f.write(f"Warmup Steps: 1400\n")
    f.write(f"Total Training Steps: {len(train_loader) * num_epochs}\n")
    f.write(f"Teacher Model Parameters: {teacher_params:.2f}M\n")
    f.write(f"Student Model Parameters: {student_params:.2f}M\n")
    f.write(f"{'='*60}\n\n")

# Early stopping parameters
epochs_no_improve = 0
patience = 3

for epoch in range(num_epochs):
    train_loss = distill_train(
        teacher_model=teacher,
        student_model=student,
        dataloader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epoch=epoch,
    )
    eval_loss = eval(
        model=student,
        eval_loader=eval_loader,
        device=device
    )
    training_loss.append(train_loss)
    evaluation_loss.append(eval_loss)
    train_perplexity = torch.exp(torch.tensor(train_loss))
    eval_perplexity = torch.exp(torch.tensor(eval_loss))

    # Log the losses and perplexities for each epoch
    with open(log_file, "a") as f:
        f.write(f"{'='*60}\n")
        f.write(f"Epoch {epoch+1}\n")
        f.write(f"Train Loss: {train_loss:.4f} | Train Perplexity: {train_perplexity:.4f}\n")
        f.write(f"Eval  Loss: {eval_loss:.4f} | Eval  Perplexity: {eval_perplexity:.4f}\n")
        f.write(f"{'='*60}\n\n")

    # Early stopping logic
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        torch.save(student.state_dict(), last_checkpoint)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            with open(log_file, "a") as f:
                f.write(f"Early stopping triggered at epoch {epoch}\n")
            break

# Final evaluation on the test set
test_loss = eval(student, test_loader, device)
test_ppl = math.exp(test_loss)

# Log final test results

with open(log_file, "a") as f:
    f.write(f"{'='*60}\n")
    f.write(f"Final Test Results\n")
    f.write(f"Test Loss: {test_loss:.4f} | Test Perplexity: {test_ppl:.4f}\n")
    f.write(f"{'='*60}\n")

# Plotting loss curves
loss_path = os.path.join(log_dir, f"{model_name}_loss.png")
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss')
plt.plot(range(1, len(evaluation_loss) + 1), evaluation_loss, label='Evaluation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss')
plt.legend()
plt.grid(True)
plt.savefig(loss_path)

# Plotting perplexity curves
perplexity_path = os.path.join(log_dir, f"{model_name}_perplexity.png")
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