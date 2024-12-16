import torch
from torch.utils.data import Dataset
from pathlib import Path
from transformers import PretrainedConfig, BertConfig, BertForPreTraining
from transformers import Trainer, TrainingArguments
import glob

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = glob.glob(str(Path(data_dir) / "*.pt"))
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load the .pt file
        data = torch.load(self.file_paths[idx])
        # Assuming shape is (1, n, 4096), squeeze the first dimension
        data = data.squeeze(0)
        
        return {
            "input_ids": data,
            "labels": data.clone()
        }

def main():
    # 1. Define model configuration
    config = BertConfig(
        hidden_size=4096,
        num_hidden_layers=12,
        num_attention_heads=16,
        intermediate_size=4096 * 4,
        max_position_embeddings=512,
        vocab_size=5  # Modified to use vocab size of 5
    )
    
    # 2. Initialize model
    model = BertForPreTraining(config)
    
    # 3. Create datasets
    train_dataset = CustomDataset("path/to/train/data")
    eval_dataset = CustomDataset("path/to/eval/data")
    
    # 4. Define training arguments
    training_args = TrainingArguments(
        output_dir="./custom-transformer",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        max_grad_norm=1.0
    )
    
    # 5. Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # 6. Train the model
    trainer.train()
    
    # 7. Save the final model
    trainer.save_model("./custom-transformer-final")

if __name__ == "__main__":
    main()