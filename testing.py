import pickle
with open("multi-label-binarizer.pkl", "wb") as f:
  pickle.dump(multilabel, f)

!zip -r distilbert.zip "/content/distilbert-finetuned-sdg-multi-label"

text="""  """
encoding = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
encoding.to(trainer.model.device)

outputs = trainer.model(**encoding)

sigmoid = torch.nn.Sigmoid()
probs = sigmoid(outputs.logits[0].cpu())
preds = np.zeros(probs.shape)
preds[np.where(probs>=0.3)] = 1

multilabel.classes_

multilabel.inverse_transform(preds.reshape(1,-1))

model.save_pretrained("./distilbert-finetuned-sdg-multi-label")
tokenizer.save_pretrained("./distilbert-finetuned-sdg-multi-label")