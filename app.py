import pandas as pd
df = pd.read_csv("spam_text.csv")

X=df['Message']
Y=df['Category']

from sklearn.feature_extraction.text import CountVectorizer
X_CV = CountVectorizer().fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_CV, Y)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

model = LogisticRegression()
model.fit(X_train, Y_train)
pred = model.predict(X_test)
cm=confusion_matrix(Y_test, pred)
print(cm)
acc=(cm[0,0]+cm[1,1]+cm[2,2])/sum(sum(cm))
print('Accuracy without TFIDF is ', acc)