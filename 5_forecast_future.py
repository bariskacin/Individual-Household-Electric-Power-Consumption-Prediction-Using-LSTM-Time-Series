import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import pandas as pd

# model ve scaler'ı yükle
model = load_model("lstm_model.h5") # eğitilmiş LSTM modelini yükle
scaler = joblib.load("scaler.save") # normalizasyon sırasında kullanılan scaler'ı yükle

# zaman serisi verisini yükle
df_hourly = pd.read_csv("df_hourly.csv", index_col=0, parse_dates=True)

# son 48 saati al, 24 saatlik geçmişi kullanarak sonraki 24 saati tahmin et
last_48 = df_hourly.iloc[-48:].copy() # son 48 saati al
last_24_real = last_48.iloc[24:].values # ilk 24 saat, modelin girdisi olacak
real_next_24 = last_48.iloc[24:].values # son 24 saat, model tahminleri ile karşılaştırmak için gerçek değerler

# normalize edilmiş veriyi al
X_test = np.load("X_test.npy")
forecast_input = X_test[-1].copy() # test setinin son örneğini al (24, 1)

# model ile 24 saatlik tahmin yap
future_predictions = [] # tahminleri saklamak için liste
for _ in range(24):
    input_3d = forecast_input.reshape(1, forecast_input.shape[0], forecast_input.shape[1]) # (örnek sayısı, zaman adımı, özellik sayısı)
    next_scaled = model.predict(input_3d, verbose=0)[0] # ölçeklenmiş tahmin -> 0-1 aralığında
    next_value = scaler.inverse_transform(next_scaled.reshape(1, -1))[0, 0] # orijinal ölçeğe döndür
    future_predictions.append(next_value) # tahmini değerini listeye ekle
    # yeni girdi penceresini güncelle, ilk değer at, tahmin edilen değeri sona ekle
    forecast_input = np.vstack((forecast_input[1:], next_scaled.reshape(1, 1)))

# karşılaştırmalı grafik çiz
plt.figure()
plt.plot(real_next_24.flatten(), label="Gerçek Değerler (Gelecek 24 Saat)", linewidth=2)
plt.plot(future_predictions, label = "Tahmin Edilen Değerler (Gelecek 24 Saat)", linestyle="--", linewidth=2)
plt.title("LSTM Modeli ile Gelecek 24 Saatlik Enerji Tüketimi Tahmini")
plt.xlabel("Saat")
plt.ylabel("Global Aktif Güç (kilowatt)")
plt.legend()
plt.grid()
plt.show()