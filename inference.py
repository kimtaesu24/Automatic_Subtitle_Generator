import torch
from transformers import BartForConditionalGeneration, BartTokenizer


# Load the BART model and tokenizer
model_name = 'facebook/bart-base'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.load_state_dict(torch.load("pretrained_model/1000epochs.pt"))
model.to(device)
model.eval()

sentence = 'some people are sitting in front of the computer'
emotion = 'sad'
prompt = "Rewrite '" + sentence + "' to convey a '" + emotion + "' emotion"

input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
output_ids = model.generate(input_ids)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)