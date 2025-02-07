from sklearn.datasets import load_breast_cancer
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# (1) veri seti incelemesi
cancer = load_breast_cancer()

df = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
df["target"] = cancer.target

# (2) ML modelinin secilmesi : KNN
X = cancer.data # features
y = cancer.target # target

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 42)

# olceklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# (3) Modelin train edilmesi
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train) # bu fonksiyon veriyi (samples + targets) kullanarak knn algosunu egitir

# (4) Sonuclarin degerlendirilmesi: test
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("accuracy=", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("confusion=", conf_matrix)

# (5) Hyperparameter ayarlanmasi
"""
    KNN: Hyperparameter = K
        K: 1,2,3, ... N
        Accuracy: %A, %B, %C, ...
"""
accuracy_values = []
k_values = []
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)
    
plt.Figure()
plt.plot(k_values, accuracy_values, marker="o" )
plt.grid()
plt.show()
    
    