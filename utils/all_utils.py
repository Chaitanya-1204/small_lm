# To make other scripts importable
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Import necessary libraries
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast ,  GPT2Config , GPT2LMHeadModel
import torch.nn.functional as F

# Import custom modules
from utils.data import BabyLmDataset , collate_fn


torch.set_float32_matmul_precision("high")



def train(model, dataloader, optimizer, device, epoch , scheduler):
    
    '''Train the model for one epoch.
        Args: 
            model: The model to train.
            dataloader: DataLoader for the training data.
            optimizer: Optimizer for updating model parameters.
            device: Device to run the model on (CPU or GPU).
            epoch: Current epoch number.
            scheduler: Learning rate scheduler.
            
        Returns:
            avg_loss: Average loss for the epoch.
            
        '''
    # Set the model to training mode
    model.train()
    
    # Initialize variables to track total loss and tokens processed
    total_loss = 0.0
    total_tokens = 0
    
    # Create a progress bar for the training loop
    loop = tqdm(dataloader, desc=f"Epoch {epoch + 1} : ", leave=False)
    
    for batch in loop:
        # Move input data to the specified device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.autocast(device_type = "cuda" , dtype  = torch.bfloat16):
             
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        
        # Update the progress bar with the current loss and learning rate
        current_lr = scheduler.get_last_lr()[0]
        loop.set_postfix(loss=loss.item(), lr=current_lr)
        
        # Calculate the number of active tokens in the labels
        active_tokens = (labels != -100).sum().item()

        # Accumulate the loss and tokens processed
        total_loss += loss.item() * active_tokens
        total_tokens += active_tokens
    
    # Calculate the average loss for the epoch
    avg_loss = total_loss / total_tokens
    
    
    return avg_loss


def eval(model, dataloader, device):
    
    '''
        Evaluate the model on the validation or test set.
        Args:
            model: The model to evaluate.
            dataloader: DataLoader for the evaluation data.
            device: Device to run the model on (CPU or GPU).
        Returns:
            avg_loss: Average loss for the evaluation set.
    
    '''
    
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize variables to track total loss and tokens processed
    total_loss = 0.0
    total_tokens = 0
    
    # Create a progress bar for the evaluation loop
    loop = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        
        for batch in loop:
            
            # Move input data to the specified device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision
            with torch.autocast(device_type = device , dtype  = torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                
            # Update the progress bar with the current loss
            loop.set_postfix(loss=loss.item())
            
            # Calculate the number of active tokens in the labels
            active_tokens = (labels != -100).sum().item()
               
             # Accumulate the loss and tokens processed   
            total_loss += loss.item() * active_tokens
            total_tokens += active_tokens

    # Calculate the average loss for the evaluation set
    avg_loss = total_loss / total_tokens
    
    
    return avg_loss


def get_dataloaders(batch_size , num_workers , seq_length, tokenizer , train_path , pad_token_id=0 , ):
    
    ''' 
        Get DataLoaders for training, evaluation, and testing datasets.
        
        Args:
            batch_size: Batch size for the DataLoaders.
            num_workers: Number of worker threads for DataLoader.
            seq_length: Maximum sequence length for tokenization.
            tokenizer: Tokenizer to use for encoding text.
            train_path: Path to the training dataset.
            pad_token_id: ID of the padding token (default is 0).
        
        Returns:
            train_loader: DataLoader for the training dataset.  
            eval_loader: DataLoader for the evaluation dataset.
            test_loader: DataLoader for the testing dataset.
        
        '''
    

    # Loading Datasets 
    train_dataset = BabyLmDataset(train_path , seq_length , tokenizer ,split = "train" ,  random_chunk=True)

    eval_path = os.path.join("cleaned_datasets" , "dev")
    eval_dataset = BabyLmDataset(eval_path , seq_length , tokenizer , split = "dev", offset=0, random_chunk=False) 

    test_path = os.path.join("cleaned_datasets" , "test")
    test_dataset = BabyLmDataset(test_path , seq_length , tokenizer , split = "test", offset=0, random_chunk=False) 
    
    # Building DataLoaders

    train_loader = DataLoader(
        train_dataset,
        batch_size= batch_size ,
        shuffle=True,
        num_workers = num_workers, 
        pin_memory = True,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=pad_token_id)
    )

    eval_loader = DataLoader(
        eval_dataset , 
        batch_size = batch_size, 
        shuffle = False ,
        num_workers = num_workers , 
        pin_memory = True, 
    
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=pad_token_id)
        
    )

    test_loader = DataLoader(
        test_dataset , 
        batch_size = batch_size, 
        shuffle = False , 
        num_workers = num_workers , 
        pin_memory = True,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=pad_token_id)
        
    )

    return train_loader, eval_loader, test_loader
        

def get_tokenizer(tokenizer_path, seq_length):
    
    '''
        Get the tokenizer for the model.
        
        Args:
            tokenizer_path: Path to the tokenizer file.
            seq_length: Maximum sequence length for tokenization.
        
        Returns:
            tokenizer: The initialized tokenizer.
    
    '''
    
    print(tokenizer_path)
  
    # Load the tokenizer from the specified path
    tokenizer = GPT2TokenizerFast(tokenizer_file=tokenizer_path)
    
    # Set special tokens and model configuration
    tokenizer.pad_token = "<pad>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    
    # Set the maximum sequence length for the tokenizer
    tokenizer.model_max_length = seq_length
    
    # Ensure the tokenizer has a padding token ID
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0  
    
    return tokenizer

def get_model(config , log_file, last_checkpoint , device):
    
    '''
        Get the GPT-2 model and its configuration.
        
        Args:
            tokenizer: Tokenizer to use for the model.
            log_file: Path to the log file for logging model information.
            last_checkpoint: Path to the last checkpoint file (if it exists).
            device: Device to run the model on (CPU or GPU).
        
        Returns:
            model: The initialized GPT-2 model.
            config: The configuration of the GPT-2 model.
    
    '''
    
    
    
    # Initialize the model
    model = GPT2LMHeadModel(config)
    
    
    # Load the last checkpoint if it exists
    if os.path.exists(last_checkpoint):
        with open(log_file, "a") as f:
            f.write(f"Loading model weights from {last_checkpoint}\n")
            
        model.load_state_dict(torch.load(last_checkpoint))
    # Else initialize a new model
    else:
        with open(log_file, "a") as f:
            f.write("Initializing new model\n")
    
    # Move the model to the specified device
    model = model.to(device)
    model.gradient_checkpointing_enable() # Enable gradient checkpointing for memory efficiency

    # Log the number of parameters in the model
    with open(log_file, "a") as f:
        f.write(f"Model Parameters: {model.num_parameters() / 1e6:.2f}M\n")
        
    return model

def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    # Resize teacher logits to match student shape if needed
    teacher_soft = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    
    # KLD loss between student and teacher soft logits
    kld_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean', log_target=True) * (temperature ** 2)

    # Normal cross-entropy loss
    ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)),
                              labels.view(-1),
                              ignore_index=-100)

    return alpha * ce_loss + (1 - alpha) * kld_loss




def distill_train(teacher_model, student_model , dataloader, optimizer, device, epoch , scheduler):
    
    '''Train the model for one epoch.
        Args: 
            model: The model to train.
            dataloader: DataLoader for the training data.
            optimizer: Optimizer for updating model parameters.
            device: Device to run the model on (CPU or GPU).
            epoch: Current epoch number.
            scheduler: Learning rate scheduler.
            
        Returns:
            avg_loss: Average loss for the epoch.
            
        '''
    # Set the model to training mode
    student_model.train()
    
    # Initialize variables to track total loss and tokens processed
    total_loss = 0.0
    total_tokens = 0
    
    # Create a progress bar for the training loop
    loop = tqdm(dataloader, desc=f"Epoch {epoch + 1} : ", leave=False)
    
    for batch in loop:
        # Move input data to the specified device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.autocast(device_type = "cuda" , dtype  = torch.bfloat16):
             
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = distillation_loss(
                        student_outputs.logits, 
                        teacher_outputs.logits, 
                        labels,
                        temperature=2.0, 
                        alpha=0.5)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        
        # Update the progress bar with the current loss and learning rate
        current_lr = scheduler.get_last_lr()[0]
        loop.set_postfix(loss=loss.item(), lr=current_lr)
        
        # Calculate the number of active tokens in the labels
        active_tokens = (labels != -100).sum().item()

        # Accumulate the loss and tokens processed
        total_loss += loss.item() * active_tokens
        total_tokens += active_tokens
    
    # Calculate the average loss for the epoch
    avg_loss = total_loss / total_tokens
    
    
    return avg_loss
