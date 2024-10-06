import torch
from transformers import BertTokenizer, BertForTokenClassification
import argparse

model_path = './models/bert_sanskrit/'
model = BertForTokenClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

id2label = model.config.id2label


def predict(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", is_split_into_words=True)
    outputs = model(**inputs).logits
    predictions = outputs.argmax(dim=-1)

    predicted_labels = [id2label[p.item()] for p in predictions[0]]
    return predicted_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, required=True, help="Sanskrit sentence for inference.")
    args = parser.parse_args()

    sentence = args.sentence
    labels = predict(sentence)
    print(f"Labels: {labels}")
