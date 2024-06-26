import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import pandas as pd

# Loading pre-trained model and tokenizer
model_name = "bert-base-uncased"            # I think the model is enough for the task
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_name)

df = pd.read_csv('data\historical_data.csv')
# Preparing data for training
# `df` was set by running "data_collection.ipynb" file.

def prepare_data(df):
    # Converting 'df' to sequences for training
    sequences = []
    labels = []
    window_size = 24  # using 24 hours data to predict next hour

    for i in range(len(df) - window_size):
        sequences.append(df.iloc[i:i+window_size].values.tolist())
        labels.append(df.iloc[i+window_size]['close'])

    return sequences, labels

sequences, labels = prepare_data(df)

# Split data into training and test sets
train_size = int(0.8 * len(sequences))
train_sequences, test_sequences = sequences[:train_size], sequences[train_size:]
train_labels, test_labels = labels[:train_size], labels[train_size:]

class BTCPriceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.sequences[idx], dtype=torch.float),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

train_dataset = BTCPriceDataset(train_sequences, train_labels)
test_dataset = BTCPriceDataset(test_sequences, test_labels)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
