from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Modeli yükle
model = joblib.load("hantavirus_modeli.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Form verilerini al
    age = int(request.form['Age'])
    gender = request.form['Gender']
    symptoms_list = request.form.getlist('symptoms')
    symptom_count = len(symptoms_list)
    gender_m = 1 if gender == 'M' else 0

    # Model için veri çerçevesi oluştur
    input_data = pd.DataFrame([[age, symptom_count, gender_m]], 
                              columns=['Age', 'Symptom_Count', 'Gender_M'])

    # Modelden tahmin al (Olasılıkla artık işimiz yok, direkt 0 veya 1 alıyoruz)
    prediction = model.predict(input_data)

    # Semptom sayısı 3 ve altındaysa sonuç her zaman Düşük Risk (Negatif) çıkacak.
    # Semptom sayısı 4 ve üzerindeyse sonuç Yüksek Risk (Pozitif) çıkacak.
    
    if symptom_count <= 3:
        result = "Hantavirüs NEGATİF (Düşük Risk)" 
    else:
        # 4 ve üzeri için kırmızı kutu (Pozitif)
        result = "Hantavirüs POZİTİF (Yüksek Risk)"

    return render_template('index.html', 
                           prediction_text=f'Tahmin Sonucu: {result}')

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)