# Importiere Bibliotheken
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Datensatz Iris-Blumen
data = load_iris()
X = data.data  # Merkmale
y = data.target  # Zielvariablen (Blumenarten)

# Daten in Trainings- und Testmenge aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Modell trainieren (Logistische Regression)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

#Vorhersagen treffen
y_pred = model.predict(X_test)

#Genauigkeit ausgeben
print("Genauigkeit:", accuracy_score(y_test, y_pred))
