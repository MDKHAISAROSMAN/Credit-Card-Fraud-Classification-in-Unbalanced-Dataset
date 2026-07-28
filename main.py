import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix , ConfusionMatrixDisplay , classification_report
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

data = fetch_openml("creditcard", version=1, as_frame=True)

df = data.frame
label = df.pop("Class")

x = df.to_numpy().copy()
y = label.to_numpy().copy().astype(int)


x_train,x_test,y_train,y_test = train_test_split(x,y,stratify = y,test_size =0.25,random_state=27)
scale = StandardScaler()
x_train_scaled = scale.fit_transform(x_train)
x_test_scaled = scale.fit_transform(x_test)
 
model = LogisticRegression(max_iter = 500,class_weight = "balanced")
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Greens", values_format="d")
plt.title("Confusion Matrix")
plt.show()

print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))