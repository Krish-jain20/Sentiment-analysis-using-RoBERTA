!pip install transformers datasets accelerate -q

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
import evaluate

# 1. Load dataset (Amazon Polarity)
dataset = load_dataset("amazon_polarity")

# 2. Reduce dataset for FAST training
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(20000))
dataset["test"]  = dataset["test"].shuffle(seed=42).select(range(5000))

# 3. Load tokenizer & model
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4. Tokenize
def tokenize(batch):
    return tokenizer(batch['content'], truncation=True, padding=False)

tokenized = dataset.map(tokenize, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 5. Metrics
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return metric.compute(predictions=preds, references=labels)

# 6. Training args (FAST)
training_args = TrainingArguments(
    output_dir="fast-roberta",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    logging_steps=50,
    save_strategy="no",
    eval_strategy="epoch",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("sentiment-roberta-model")
tokenizer.save_pretrained("sentiment-roberta-model")
from transformers import pipeline

model_path = "sentiment-roberta-model"

sentiment_analyzer = pipeline(
    "text-classification",
    model=model_path,
    tokenizer=model_path,
    return_all_scores=False
)
sentences = [
    "I love this product! It works perfectly.",
    "This is the worst thing I ever bought.",
    "Delivery was late but the item is good."
]

for s in sentences:
    print(s, "→", sentiment_analyzer(s)[0])
