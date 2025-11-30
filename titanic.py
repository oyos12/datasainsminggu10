import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

data = pd.read_csv("titanic.csv")
df = pd.DataFrame(data)

# Pra-pemrosesan data
df.fillna({'age': df['age'].median(), 'fare': df['fare'].median()}, inplace=True)
le = LabelEncoder()
df['encodedsex'] = le.fit_transform(df['sex']) #Ubah gender male/female jadi angka 0/1, untuk keperluan model ML

#Training model KNN
features = ['pclass', 'encodedsex', 'age', 'fare']
X = df[features]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

print(f'accuracy: {accuracy_score(y_test, y_pred):.2f}')
print('Laporan Clasification:')
print(classification_report(y_test, y_pred))
print('Matrix Konfusi:')
print(confusion_matrix(y_test, y_pred))
print()

# Model Permutation feature importance 
perm_importance = permutation_importance(knn, X_test_scaled, y_test, n_repeats=10, random_state=42)
importance_df = pd.DataFrame({
    'Variable': features,
    'Permutasian': perm_importance.importances_mean
}).sort_values(by='Permutasian', ascending=False)

print('===== Table Mutasi Data =====')
print(importance_df)
print()

print(f'Faktor-faktor yang paling mempengaruhi terhadap kemungkinan selamat yaitu {importance_df.iloc[0]["Variable"]}')
print()

#Inputan yang digunakan untuk prediksi manual
input_pclass = int(input("Kelas Sewa Titanic (1, 2, atau 3): "))
input_sex = input("Jenis kelamin Penumpang Titanic (male/female): ")
input_age = int(input("Usia penumpang: "))
input_fare = int(input("Fare (harga tiket titanic): "))
print()

#Prediksi dari Analisis data Pclass, Age, Sex dan Fare
try:
    inputencodedsex = le.transform([input_sex])[0]
    input_data = np.array([[input_pclass, inputencodedsex, input_age, input_fare]])
    input_data_scaled = scaler.transform(input_data)
    prediction = knn.predict(input_data_scaled)
except ValueError: #Dipakaikan Catcher untuk mendeteksi error, agar output yang dikeluarkan terminal lebih rapi dan stabil
    print("DATA YANG DIINPUT TIDAK BENAR!")

print("===== Prediksi dari Beberapa data kolom Pclass, Age, Sex dan Fare =====")
if prediction[0] == 1:
    print("Prediksi: Prediksi Penumpang SELAMAT.")
else:
    print("Prediksi: Prediksi Penumpang TIDAK SELAMAT.")
print()

#Prediksi Berdasarkan Analisis dari data Pclass dan Fare
input_data3 = np.array([[input_pclass, input_fare]])
input_data3_scaled = scaler.transform(np.hstack((input_data3[:, :1], np.zeros((1,2)), input_data3[:, 1:])))
prediction3 = knn.predict(input_data3_scaled)

print("===== Prediksi Berdasarkan Analisis dari data Pclass dan Fare =====")
if prediction3[0] == 1:
    print("Prediksi status penumpang: Prediksi Penumpang SELAMAT.")
else:
    print("Prediksi status penumpang: Prediksi Penumpang TIDAK SELAMAT.")
print()

#Prediksi Berdasarkan Analisis dari data Sex, Age dan Fare
inputencodedsex = le.transform([input_sex])[0]
input_data2 = np.array([[inputencodedsex, input_age, input_fare]])
input_data2_scaled = scaler.transform(np.hstack((np.zeros((1,1)), input_data2)))
prediction2 = knn.predict(input_data2_scaled)

print("===== Prediksi Berdasarkan Analisis dari data Sex, Age dan Fare =====")
if prediction2[0] == 1:
    print("Prediksi status penumpang: Kemungkinan SELAMAT.")
else:
    print("Prediksi status penumpang: Kemungkinan TIDAK SELAMAT.")
print()