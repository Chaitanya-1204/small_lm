import os
from pathlib import Path
from random import randrange
from tqdm import tqdm

import torch 
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class BabyLmDataset(Dataset):
    
    def __init__(self , data_dir , seq_length , tokenzier , split , offset = 0 , random_chunk = False ):
        
        super(BabyLmDataset , self).__init__()
        self.seq_length = seq_length
        self.offset = offset
        self.random_chunk = random_chunk
        
        tokenzier_name = tokenzier.__class__.__name__
        tokenized_file = Path(os.path.join(data_dir , f"tokenized_{tokenzier_name}.pt"))
        
        
        if tokenized_file.exists():
            
            print(f"Loading tokenized dataset from {tokenized_file}")
            self.data = torch.load(tokenized_file)
            
        else:
            print(f"Tokenizing dataset and saviing to {tokenized_file}")
            data = []
            
            if split == "train":
                src_files = [os.path.join(data_dir , f) for f in os.listdir(data_dir) if f.endswith(".train")]
            elif split =="dev":
                src_files = [os.path.join(data_dir , f) for f in os.listdir(data_dir) if f.endswith(".dev")]
            elif split == "test":
                src_files = [os.path.join(data_dir , f) for f in os.listdir(data_dir) if f.endswith(".test")]
            
            print(f"Tokenizing {len(src_files)} files")
            
            for f in tqdm(src_files , desc = "Tokenizing "):
                text = Path(f).read_text(encoding='utf-8')
                encoded = tokenzier.encode(text)
                print(f"Tokenized {len(encoded)} tokens from {f}")
                data.extend(encoded)

            self.data = torch.tensor(data)
                 
            print(f"Saving tokenized dataset to {tokenized_file}")
            torch.save(self.data, tokenized_file)
        
    def __len__(self):
        
        if self.random_chunk:
            return len(self.data) // self.seq_length - 1
        else:
            return (len(self.data) - self.offset) // self.seq_length
        
        
    def __getitem__(self, idx):
        # Pick a random sequence length (between 16 and self.seq_length)
        actual_seq_len = randrange(16, self.seq_length + 1)
        
        # Determine the max valid start index
        max_start = len(self.data) - actual_seq_len
        if max_start <= 0:
            raise IndexError("Dataset too small for selected sequence length")

        # Compute start index
        if self.random_chunk:
            start = randrange(0, max_start)
        else:
            start = self.offset + idx * self.seq_length
            if start > max_start:
                start = max_start

        end = start + actual_seq_len
        chunk = self.data[start:end]

        return {
            "input_ids": chunk.long(),
            "labels": chunk.long(),
            "attention_mask": torch.ones(chunk.size(0), dtype=torch.long)
        }
                
                
                            
            
            
def collate_fn(batch , pad_token_id = 0):
    
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]

    # Pad input_ids and attention_masks with pad_token_id (usually 0)
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # Pad labels with -100 so loss ignores them
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded
    }