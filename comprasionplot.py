import matplotlib.pyplot as plt

models = [
    "Multinomial NB",
    "Stochastic Gradient Descent",
    "Logistic Regression",
    "Linear SVC",
    "DistilBERT"
]
accuracies = [0.7642, 0.8579, 0.7299, 0.8538, 0.92]

# Plotting the bar graph
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'coral', 'lightpink', 'gold'])

# Adding labels and title
plt.xlabel("Machine Learning Models")
plt.ylabel("Jaccard Score")
plt.title("Jaccard Score Comparison of Different ML Models for SDG Classification")

for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()