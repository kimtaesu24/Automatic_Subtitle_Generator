import pickle
import torch
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.utils.data import DataLoader, Dataset

# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Prepare your dataset
with open('dataset.pickle', 'rb') as f:
    pair = pickle.load(f)
    print(len(pair))
    
examples = [("Translate 'Hello' to French.", "Bonjour"), ("Translate 'Goodbye' to French.", "Au revoir")]
# dataset = CustomDataset(examples)
dataset = CustomDataset(pair)

# Load the BART model and tokenizer
model_name = 'facebook/bart-base'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Fine-tuning settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
epochs = 1000
learning_rate = 1e-5

# Data loading
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
model.to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

pbar = tqdm(range(epochs), position=0, leave=False, desc='epoch')
for epoch in pbar:
    for batch in tqdm(dataloader, position=1, leave=False, desc='batch'):
        input_batch = tokenizer(batch[0], truncation=True, padding=True, return_tensors='pt').to(device)
        labels_batch = tokenizer(batch[1], truncation=True, padding=True, return_tensors='pt').to(device)

        outputs = model(
            input_ids=input_batch['input_ids'],
            attention_mask=input_batch['attention_mask'],
            labels=labels_batch['input_ids'],
            decoder_attention_mask=labels_batch['attention_mask']
        )

        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), '/home2/s20235100/PDL/pretrained_model/'+str(epoch + 1)+'epochs.pt')



# Example usage after fine-tuning
sentence = 'some people are sitting in front of the computer'
emotion = 'sad'
prompt = "Rewrite '" + sentence + "' to convey a '" + emotion + "' emotion"
input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
output_ids = model.generate(input_ids)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)