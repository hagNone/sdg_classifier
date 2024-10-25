!pip install -U accelerate
!pip install -U transformers

import pandas as pd

data = pd.read_excel("/content/drive/MyDrive/Classroom/AIML Lab 2024/Group 6&7-Classification.xlsx")
data.head()

data.info()

data.duplicated().sum()

data['Overview'].str.len().plot.hist(bins=50)

data['SDG'] = data['SDG'].str.replace(', ', ',')
data['SDG'] = data['SDG'].str.split(',')

sdg_counts = [s for sdg in data['SDG'] for s in sdg]
sdg_counts

from sklearn.preprocessing import MultiLabelBinarizer
multilabel = MultiLabelBinarizer()

labels = multilabel.fit_transform(data['SDG']).astype('float32')
texts = data["Overview"].tolist()
labels, texts

import torch
from transformers import DistilBertTokenizer, AutoTokenizer
from transformers import DistilBertForSequenceClassification, AutoModelForSequenceClassification

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

checkpoint = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)
model = DistilBertForSequenceClassification.from_pretrained(checkpoint, num_labels=len(labels[0]), problem_type="multi_label_classification")

class CustomDataset(Dataset):
  def __init__(self, texts, labels, tokenizer, max_len=512):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = str(self.texts[idx])
    label = torch.tensor(self.labels[idx])

    encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')

    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'label': label
    }
  
train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer)



import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss
from transformers import EvalPrediction
import torch

def multi_labels_metrics(predictions, labels, threshold=0.3):
  sigmoid = torch.nn.Sigmoid()
  probs = sigmoid(torch.Tensor(predictions))

  y_pred = np.zeros(probs.shape)
  y_pred[np.where(probs>threshold)] = 1
  y_true = labels

  f1 = f1_score(y_true, y_pred, average='macro')
  roc_auc = roc_auc_score(y_true, y_pred, average='macro')
  hamming = hamming_loss(y_true, y_pred)

  metrics = {
      'f1': f1,
      'roc_auc': roc_auc,
      'hamming': hamming
  }

  return metrics

def compute_metrics(p: EvalPrediction):
  preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
  result = multi_labels_metrics(predictions=preds, labels=p.label_ids)
  return result

from transformers import DataCollatorWithPadding

class MultiLabelDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # Separate the labels from the other features
        labels = [feature.pop("label") for feature in features]
        batch = super().__call__(features)

        # Stack the labels
        batch["labels"] = torch.stack(labels)
        return batch

from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    output_dir='./results',
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=2
)

trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=compute_metrics, data_collator=MultiLabelDataCollator(tokenizer))

trainer.train()

trainer.evaluate()

trainer.save_model("distilbert-finetuned-sdg-multi-label")

