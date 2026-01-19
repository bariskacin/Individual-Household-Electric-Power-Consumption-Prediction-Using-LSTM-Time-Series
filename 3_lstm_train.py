import numpy as np
from tensorflow.keras.models import Sequential # base model, katmanları bunun üzerine ekleyeceğiz
from tensorflow.keras.layers import LSTM, Dense # LSTM ve dense katmanları
from tensorflow.keras.callbacks import EarlyStopping # erken durdurma için
from tensorflow.keras.losses import MeanSquaredError # kayıp fonksiyonu

import matplotlib.pyplot as plt

# veriyi yükle
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# modeli tanımla
model = Sequential() # sıralı model oluştur

# LSTM katmanı ekle
model.add(LSTM(
    units = 512,               # LSTM hücre sayısı
    activation = "tanh",      # aktivasyon fonksiyonu
    return_sequences = True, # tek katmanlı LSTM için False, çok katmanlı LSTM için True
    input_shape=(X_train.shape[1], X_train.shape[2]) # (zaman adımı, özellik sayısı)
))

# ikinci LSTM katmanı ekle
model.add(LSTM(
    units = 272,               # LSTM hücre sayısı
    activation = "tanh",      # aktivasyon fonksiyonu
    return_sequences = False   # son LSTM katmanı olduğu için False
))

# dense layer ekle
# sadece 1 saatlik enerji tahmin yapacağımız için tek bir çıktı birimi
model.add(Dense(
    units = 1,                # çıktı birimi
    activation = "linear"     # regresyon için lineer aktivasyon
))

# model compile (derleme) et
model.compile(
    optimizer ="adam",                        # optimizasyon algoritması
    loss = MeanSquaredError(),                # ortalama kare hata kaybı
    metrics = ["mae"]                         # izlenecek metrik
)

# erken durdurma callback'i oluştur
early_stop = EarlyStopping(
    monitor = "val_loss",      # doğrulama kaybını izle
    patience = 10,             # art arda eğitim tekrarında iyileşme olmazsa durdur
    restore_best_weights = True # en iyi ağırlıkları geri yükle
)

# eğitimi başlat
history = model.fit(
    X_train, y_train,                      # eğitim verisi
    validation_data = (X_test, y_test),    # doğrulama verisi
    epochs = 100,                          # eğitim tekrar sayısı
    batch_size = 32,                       # her eğitim adımında kullanılacak örnek sayısı
    callbacks = [early_stop],              # eğitim sırasında erken durdurma
    verbose = 1,                            # eğitim sürecini göster
    shuffle = False                         # zaman serisi verisi olduğu için karıştırma
)

# kayıp grafiğini çiz
plt.plot(history.history["loss"], label="Eğitim Kaybı")
plt.plot(history.history["val_loss"], label="Doğrulama Kaybı")
plt.title("Model Kayıp Grafiği")
plt.xlabel("Eğitim tekrarları (Epochs)")
plt.ylabel("Kayıp (MSE)")
plt.legend()
plt.show()

# modeli kaydet
model.save("lstm_model.h5") # eğitilmiş modeli HDF5 formatında kaydet