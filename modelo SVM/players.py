import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Carregue o conjunto de dados
data = pd.read_csv('player_data.csv')

# Seleção dos atributos relevantes (exemplo)
selected_features = ['height', 'weight', 'year_start']

# Divisão dos dados em recursos (X) e variável alvo (y)
X = data[selected_features]
y = data['position']

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criação e treinamento do modelo SVM
model = SVC()
model.fit(X_train, y_train)

# Predições no conjunto de teste
y_pred = model.predict(X_test)

# Cálculo da acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo SVM:", accuracy)
