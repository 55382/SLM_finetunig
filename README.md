
# Symptom-Based Text Generation using GPT-2

### Overview
This project uses a GPT-2 language model to generate symptom-based text. It processes a dataset containing medical conditions and their associated symptoms, tokenizes the data, and trains a transformer model to generate relevant text.

### Requirements
- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- Pandas
- tqdm

### Installation
```bash
pip install torch transformers pandas tqdm
```

### Data Preparation
The dataset consists of medical conditions and their symptoms. It is first converted into a Pandas DataFrame:
```python
updated_data = [{'Name': item['Name'], 'Symptoms': item['Symptoms']} for item in data_sample['train']]
df = pd.DataFrame(updated_data)
```

### Tokenization and Model Setup
Using `distilgpt2` from Hugging Face:
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2').to(device)
```

### Custom Dataset Class
A PyTorch `Dataset` class is implemented to structure the data:
```python
from torch.utils.data import Dataset

class LanguageDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.data = df.to_dict(orient='records')
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = f"{self.data[idx]['Name']} | {self.data[idx]['Symptoms']}"
        tokens = self.tokenizer.encode_plus(text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
        return tokens
```

### Training Setup
Splitting the dataset:
```python
from torch.utils.data import DataLoader, random_split

train_size = int(0.8 * len(data_sample))
valid_size = len(data_sample) - train_size
train_data, valid_data = random_split(data_sample, [train_size, valid_size])

dataloader_train = DataLoader(train_data, batch_size=8, shuffle=True)
dataloader_valid = DataLoader(valid_data, batch_size=8)
```

### Model Training
```python
import torch.optim as optim
import torch.nn as nn

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=5e-4)

for epoch in range(10):
    model.train()
    for batch in dataloader_train:
        optimizer.zero_grad()
        inputs = batch['input_ids'].squeeze(1).to(device)
        outputs = model(input_ids=inputs, labels=inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

### Model Evaluation
```python
model.eval()
with torch.no_grad():
    for batch in dataloader_valid:
        inputs = batch['input_ids'].squeeze(1).to(device)
        outputs = model(input_ids=inputs, labels=inputs)
        print(f"Validation Loss: {outputs.loss.item()}")
```

### Inference Example
To generate text based on symptoms:
```python
input_text = "Fever, Cough, Fatigue"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Conclusion
This project demonstrates training a GPT-2 model to understand and generate symptom-based text. It can be further fine-tuned with larger datasets and evaluated for practical medical applications.

