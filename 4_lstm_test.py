import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # eğitilmiş modeli yüklemek için
from sklearn.metrics import mean_squared_error, mean_absolute_error # hata metrikleri
import joblib # scaler'ı yüklemek için

# verileri ve modeli yükle
X_test = np.load("X_test.npy") # test verisinin giriş dizisini yükle
y_test = np.load("y_test.npy") # test verisinin çıkış dizisini yükle

model = load_model("lstm_model.h5") # eğitilmiş LSTM modelini yükle
scaler = joblib.load("scaler.save") # normalizasyon sırasında kullanılan scaler'ı yükle

# tahmin yap
# model test verisi üzerinde tahmin yap
model.predict(X_test) # modelin predict metodunu kullanarak tahmin yap
y_pred_scaled = model.predict(X_test) # ölçeklenmiş tahmin değerleri

# tahmin ve gerçek değerleri orijinal ölçeğe geri döndür
y_pred = scaler.inverse_transform(y_pred_scaled) # tahmin edilen değerleri orijinal ölçeğe döndür
y_true = scaler.inverse_transform(y_test) # gerçek değerleri orijinal ölçeğe döndür

# hata metriklerini hesapla
mae = mean_absolute_error(y_true, y_pred) # Ortalama Mutlak Hata: ne kadar sapma olduğunu gösterir
rmse = np.sqrt(mean_squared_error(y_true, y_pred)) # Kök Ortalama Kare Hata: sapmanın karekök ortalamasını gösterir

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# ilk 200 tahminin grafiğini çiz
plt.figure()
plt.plot(y_true[:200], label = "Gerçek", linewidth = 2)
plt.plot(y_pred[:200], label = "Tahmin", linestyle = "--")
plt.title("LSTM Modeli Tahminleri")
plt.xlabel("Saat")
plt.ylabel("Güç Tüketimi (kilowatt)")
plt.legend()
plt.show()