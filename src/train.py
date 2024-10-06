from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification
from data_processing import load_data, tokenize_align
from model import load_model, load_tokenizer

dataset=load_data('sanskrit_segment_dataset')
train_data=dataset['train']
val_data=dataset['validation']

label_list = ['O', 'B', 'I', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
label2id = {label: i for i, label in enumerate(label_list)}

train_dataset = train_data.map(lambda x: tokenize_align(x, label2id), batched=True)
val_dataset = val_data.map(lambda x: tokenize_align(x, label2id), batched=True)

model=load_model(label_list)
tokenizer=load_tokenizer()

data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir='./models/bert_sanskrit/',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained('./models/bert_sanskrit/')
tokenizer.save_pretrained('./models/bert_sanskrit/')