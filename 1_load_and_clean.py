import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# veri seti okuma
df = pd.read_csv(
    "household_power_consumption.txt", # veri seti
    sep=";", # ayırıcı
    # 16/12/2006 (date) ve 17:24:00 (time) formatlarını birleştirip datetime tipine çevirme
    parse_dates={"Datetime": ["Date", "Time"]},
    infer_datetime_format=True, # tarih formatını otomatik algılama
    na_values=["?"], # eksik değerler için işaretçi. "?" karakterini NaN olarak kabul et
    low_memory=False # büyük veri setlerinde bellek optimizasyonu için
)

print(df.head())

# datetime sütununu indeks olarak ayarlama
df.set_index("Datetime", inplace=True)
print(df.head())

# Global_active_power sütununu seç: "4.216" -> 4.216
df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"], # sayıya çevrilecek sütun
    errors="coerce" # dönüştürülemeyen değerleri NaN yap
)

# eksik değerleri kontrol etme
df = df.dropna(subset=["Global_active_power"]) # Global_active_power sütunundaki NaN değerleri sil

# saatlik ortalamaya göre yeniden örnekleme
df_hourly = df["Global_active_power"].resample("H").mean()
print(df_hourly.head())

# zaman serisi görselleştirme
plt.plot(df_hourly, label = "Saatlik Ortalama Global Aktif Güç")
plt.title("Enerji Tüketimi")
plt.xlabel("Zaman")
plt.ylabel("Global Aktif Güç (kilowatt)")
plt.legend()
plt.show()

# saatlik yeniden örneklenmiş veriyi kaydet
df_hourly.to_csv("df_hourly.csv")