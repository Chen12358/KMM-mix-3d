# fix_lat_descending.py
import numpy as np

sst = np.load("sst_data.npy")  # 上一步保存的 SST 数据
lat = np.load("lat.npy")
lon = np.load("lon.npy")
time = np.load("time.npy")

# 如果纬度是升序（-90→90），则改为降序（90→-90）
if lat[0] < lat[-1]:
    lat = lat[::-1]
    sst = np.flip(sst, axis=1)  # 翻转纬度轴（假设形状是 (time, lat, lon)）
    print("已将纬度改为降序 (90→-90)，并同步翻转数据。")

np.save("sst_data.npy", sst)
np.save("lat_desc.npy", lat)
print("Saved sst_data_desc.npy 和 lat_desc.npy.")
