import pickle
import warnings
warnings.filterwarnings("ignore") 
from sklearn.feature_extraction.text import TfidfVectorizer

with open("/content/drive/MyDrive/ML PROJECT/SDG_classifier.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open("/content/drive/MyDrive/ML PROJECT/tfidf_vectorizer.pkl", "rb") as f:
    TF_IDF = pickle.load(f)

with open("/content/drive/MyDrive/ML PROJECT/multilabel_binarizer.pkl", "rb") as f:
    MULTILABEL = pickle.load(f)

def predict_sdg(text_content):
    X = TF_IDF.transform([text_content])
    predictions = MODEL.predict(X)
    predicted_labels = MULTILABEL.inverse_transform(predictions)
    print(f"Predicted {len(predicted_labels[0])} SDGs out of 17 SGDs.")
    return predicted_labels[0]

predicted_sdgs_SVC = predict_sdg(text_summ_SVC)
predicted_sdgs_BERT = predict_sdg(text_summ_BERT)

print("Cloq: ", predicted_sdgs_SVC)
print("IndiaCityWalks: ", predicted_sdgs_BERT)


import matplotlib.pyplot as plt

models = [
    "Multinomial NB",
    "Stochastic Gradient Descent",
    "Logistic Regression",
    "Linear SVC",
]
accuracies = [0.7642, 0.8579, 0.7299, 0.8538]

# Plotting the bar graph
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'coral', 'lightpink', 'gold'])

# Adding labels and title
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of Different ML Models for SDG Classification")

for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()