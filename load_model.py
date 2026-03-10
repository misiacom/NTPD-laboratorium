import joblib
import numpy as np

model = joblib.load("model.joblib")

print("wczytało się")

# rekord
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# predykcja
prediction = model.predict(sample)

print("predykcja dla podanego rekordu:", prediction)