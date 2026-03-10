import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# zadanie 1
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

print(data.head())
print(data.shape)
print(data.info())

# zadanie 2

# cechy
X = data.drop('target', axis=1)
y = data['target']

# podział danych na trening i test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# utworzenie modelu
model = LogisticRegression(max_iter=1000)

# trenowanie modelu
model.fit(X_train, y_train)

# przewidywanie wyników
y_pred = model.predict(X_test)

# metryki modelu
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print("metryki:")
print("accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)

# zapis
joblib.dump(model, "model.joblib")
print("zapisało się")
