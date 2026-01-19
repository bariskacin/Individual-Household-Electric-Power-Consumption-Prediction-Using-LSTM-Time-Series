import numpy as np
from sklearn.preprocessing import MinMaxScaler # verileri ölçeklemek için (normalizasyon)
import pandas as pd
import joblib # model ve işlem nesnelerini kaydetmek ve yüklemek için

# saatlik yeniden örneklenmiş veriyi yükle
df_hourly = pd.read_csv(
    "df_hourly.csv", # saatlik yeniden örneklenmiş veri dosyası
    index_col= 0, # Datetime sütununu indeks olarak ayarla
    parse_dates= True, # Datetime sütununu datetime tipine çevir
)

# NaN değerleri temizle
df_hourly.dropna(inplace=True)

# pandas DataFrame'i numpy array formatına çevir
values = df_hourly.values.reshape(-1, 1)

# normalizasyon (0-1 aralığına ölçekleme)
scaler = MinMaxScaler() # 0-1 aralığında ölçekleme için MinMaxScaler nesnesi oluştur
scaled = scaler.fit_transform(values) # önce veriye göre min-max değerlerini hesaplıyor sonra dönüştürüyor

# ölçekleyiciyi kaydet
joblib.dump(scaler, "scaler.save") # test veya gerçek zamanlı tahminde aynı ölçekleyiciyi kullanmak için kaydet

# sliding window (kaydırmalı pencere) oluşturma
def create_sliding_windows(data, window_size = 24): # LSTM modeline girdi = geçmiş 24 saat, çıktı = 25.saat
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size]) # pencere içindeki veriler
        y.append(data[i + window_size])    # pencerenin hemen sonraki veri noktası
    return np.array(X), np.array(y)

# giriş ve çıkış verilerini oluştur
window_size = 24 # geçmiş 24 saat
X, y = create_sliding_windows(scaled, window_size)

# train ve test setlerine ayırma
split = int(len(X) * 0.8) # verinin %80'i eğitim, %20'si test için
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# şekil (shape) kontrolü
print(f"X_train shape:", X_train.shape) # (örnek sayısı, zaman adımı, özellik sayısı)
print(f"y_train shape:", y_train.shape)
print(f"X_test shape:", X_test.shape)   # (örnek sayısı, zaman adımı, özellik sayısı)
print(f"y_test shape:", y_test.shape)

# kaydet
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)