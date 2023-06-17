import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

data_base = pd.read_csv("train.csv")

# Selecionar as features relevantes
important_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = data_base[important_features]
y = data_base["Survived"]

# Tratar dados faltantes
X["Age"].fillna(X["Age"].median(), inplace=True)
X["Embarked"].fillna(X["Embarked"].mode()[0], inplace=True)

# Convertendo variáveis categóricas em numéricas
x_categ = pd.get_dummies(X)

x_train, x_test, y_train, y_test = train_test_split(x_categ, y, test_size=0.2, random_state=42)

#normalização -> A padronização, garante que algoritmos que utilizam distancias de pares de dados, como o KNN,
#tenham grande distorções nas escalas das features, o que influenciaria o resultado.
sc = StandardScaler()
x_train_norm = sc.fit_transform(x_train)
x_test_norm = sc.transform(x_test)

# Criar o modelo KNN
k = 7 # Numero de vizinhos mais proximos a ser implementado no modelo
knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', leaf_size=30, n_jobs=5)

#Usando o gridSearch para achar o numero ideal de vizinhos para o modelo
k_list = list(range(1,61))

print('')
k_values = dict(n_neighbors=k_list)
grid = GridSearchCV(knn , k_values, cv=6, scoring='accuracy')
grid.fit(pd.concat([x_train,x_test]), pd.concat([y_train,y_test]))
print('Resultado gridSearch', grid)


# Treinando o modelo
knn.fit(x_train_norm, y_train)

#Resultados:
print('')
print('-----------------------results-----------------------')
print('')
print('Num de vizinhos:', k)
print('')
#Previsões
y_prev = knn.predict(x_test_norm)
print("Previsão de sobrevivência:", y_prev)

print('')

#Acurácia do modelo
accuracy = accuracy_score(y_test, y_prev)
print("Acurácia do modelo:", accuracy)

print('')
print('----------------------------------------------')