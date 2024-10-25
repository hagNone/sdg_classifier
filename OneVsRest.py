import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.multiclass import OneVsRestClassifier

df = pd.read_excel("/content/drive/MyDrive/Group 6&7-Classification.xlsx")
df.head()

df['SDG'] = df['SDG'].str.replace(', ', ',')
df['SDG'] = df['SDG'].str.split(',')
df.info()

sdg_counts = [s for sdg in df['SDG'] for s in sdg]
pd.Series(sdg_counts).value_counts()

type(df['SDG'].iloc[0])
y = df['SDG']
y

multilabel = MultiLabelBinarizer()
y = multilabel.fit_transform(df['SDG'])
y

multilabel.classes_

pd.DataFrame(y, columns=multilabel.classes_)

tfidf = TfidfVectorizer(analyzer='word', max_features=None)
X = tfidf.fit_transform(df['Overview'])
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf, file)
X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

sgd = SGDClassifier()
lr = LogisticRegression(solver='lbfgs')
svc = LinearSVC()
nb = MultinomialNB()

def j_score(y_true, y_pred):
  jaccard = np.minimum(y_true, y_pred).sum(axis = 1)/np.maximum(y_true, y_pred).sum(axis = 1)
  return jaccard.mean()*100


def print_score(y_pred, clf):
  print("Clf: ", clf.__class__.__name__)
  print('Jacard score: {}'.format(j_score(y_test, y_pred)))
  print('----')

  for classifier in [sgd, lr, svc, nb]:
  clf = OneVsRestClassifier(classifier)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print_score(y_pred, classifier)

best_clf = OneVsRestClassifier(LinearSVC())
best_clf.fit(X_train, y_train)

model_filename = 'SDG_classifier.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(best_clf, file)

multilabel_filename = 'multilabel_binarizer.pkl'
with open(multilabel_filename, 'wb') as file:
    pickle.dump(multilabel, file)

xt = tfidf.transform([text])
clf.predict(xt)
multilabel.inverse_transform(clf.predict(xt))