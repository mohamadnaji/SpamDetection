import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load trained model weights
model.load_state_dict(torch.load('models/spam_detection.pth'))
model.eval()


# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    _, predicted_label = torch.max(probabilities, 1)
    return predicted_label.item()


while True:
    # Test with custom input by user
    user_input = input("Enter your text: ")
    sentiment = predict_sentiment(user_input)
    print("Predicted spam:", "spam" if sentiment == 1 else "ham")
