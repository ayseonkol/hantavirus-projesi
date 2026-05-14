import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import joblib

# 1. Veriyi yükle
df = pd.read_csv("hantavirus_detection_dataset.csv")

# 2. Ön işleme
df['Gender_M'] = df['Gender'].map({'M': 1, 'F': 0})
X = df[['Age', 'Symptom_Count', 'Gender_M']]
y = df['Hantavirus_Positive']

# 3. Veriyi bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. İDEAL MODEL AYARI
# max_depth=5 yaptık (ne çok sığ ne çok derin)
# min_samples_leaf=10 yaptık (yüzdelerin oynaması için esneklik tanıdık)
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=5,           
    min_samples_leaf=10,  
    random_state=42
)
model.fit(X_train, y_train) 

# 5. Skoru görme
y_pred = model.predict(X_test)
print(f"Modelin F1 Skoru: {f1_score(y_test, y_pred, average='weighted'):.4f}")

# 6. Kaydet
joblib.dump(model, "hantavirus_modeli.pkl")
print("\nModel ideal ayarlarla kaydedildi!")