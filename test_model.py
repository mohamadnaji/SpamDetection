from transformers import BertForSequenceClassification, BertTokenizer
import torch

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_state_dict(torch.load('my_model.pth', map_location=torch.device('cpu')))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

while True:
    text = input("Input a message to be checked: ")
    inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Display the prediction
    if predicted_class == 0:
        print("The message is not spam.")
    else:
        print("The message is spam.")

    if input("You want to continue? Y/N") == 'N':
        exit(0)
