from transformers import BertForTokenClassification, BertTokenizer

def load_model(label_list):
    model=BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(label_list))
    model.config.id2label = {i: label for i, label in enumerate(label_list)}
    model.config.label2id = {label: i for i, label in enumerate(label_list)}

    return model

def load_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-cased')