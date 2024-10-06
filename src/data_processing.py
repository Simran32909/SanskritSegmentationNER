from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def tokenize_align(examples, label2id):
    tokenized_inputs=tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    labels=[]
    for i, label in enumerate(examples['segmentation_labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(label2id[label[word_id]])
            else:
                label_ids.append(label2id[label[word_id]])
            previous_word_id = word_id
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def load_data(dataset):
    from datasets import load_dataset
    data=load_dataset(dataset)
    return data