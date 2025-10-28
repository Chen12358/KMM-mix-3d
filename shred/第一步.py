# convert_sst_nc_to_npy_standardize.py
import numpy as np
from netCDF4 import Dataset

file_path = "sst.week.mean.nc"  # 改成你的路径
out_data  = "sst_data.npy"
out_lat   = "lat.npy"
out_lon   = "lon.npy"
out_time  = "time.npy"

def filled_array(var):
    """读取变量，填充缺测为 NaN，并返回 ndarray"""
    arr = var[:]
    if isinstance(arr, np.ma.MaskedArray):
        arr = np.ma.filled(arr, np.nan)
    return np.asarray(arr)

def to_float32(a):
    return a.astype(np.float32, copy=False)

with Dataset(file_path, "r") as ds:
    print("Variables:", list(ds.variables.keys()))

    # 1) 找到 SST 变量（默认 'sst'，否则挑第一个三维变量）
    if "sst" in ds.variables:
        v = ds.variables["sst"]
    else:
        # 自动识别第一个维度数为3的变量
        candidates = [name for name, vv in ds.variables.items() if len(vv.dimensions) == 3]
        if not candidates:
            raise RuntimeError("找不到三维的主场变量（如 'sst'）。")
        v = ds.variables[candidates[0]]
        print(f"未找到 'sst'，使用变量: {v.name}")

    # 2) 读取原始坐标变量（名字可能是 lat/latitude、lon/longitude）
    #    尽量从 sst 的维度中定位坐标名
    dims = v.dimensions  # e.g. ('time','lat','lon')
    dim_map = {d.lower(): d for d in dims}

    def get_var_by_names(candidates):
        for name in candidates:
            if name in ds.variables:
                return ds.variables[name]
        return None

    # 优先从维度名匹配
    time_var = ds.variables[dim_map.get("time")] if "time" in dim_map else get_var_by_names(["time"])
    lat_var  = ds.variables[dim_map.get("lat")]  if "lat"  in dim_map else get_var_by_names(["lat","latitude","y"])
    lon_var  = ds.variables[dim_map.get("lon")]  if "lon"  in dim_map else get_var_by_names(["lon","longitude","x"])

    if time_var is None or lat_var is None or lon_var is None:
        # 兜底：在所有变量里找可能的坐标
        if time_var is None:
            time_var = get_var_by_names(["time"])
        if lat_var is None:
            lat_var = get_var_by_names(["lat","latitude","y"])
        if lon_var is None:
            lon_var = get_var_by_names(["lon","longitude","x"])
    if time_var is None or lat_var is None or lon_var is None:
        raise RuntimeError("无法识别 time/lat/lon 坐标变量。")

    # 3) 开启自动掩码缩放（scale_factor/add_offset）
    v.set_auto_maskandscale(True)

    # 4) 读取数据与坐标
    sst  = filled_array(v)
    time = filled_array(time_var)
    lat  = filled_array(lat_var)
    lon  = filled_array(lon_var)

    # 记录维度顺序，确保 sst 轴映射正确
    dim_to_axis = {d: i for i, d in enumerate(v.dimensions)}
    t_ax = dim_to_axis[time_var.name]
    la_ax = dim_to_axis[lat_var.name]
    lo_ax = dim_to_axis[lon_var.name]

    print(f"SST shape (raw): {sst.shape}, dims: {v.dimensions}")
    print(f"time size: {time.shape}, lat size: {lat.shape}, lon size: {lon.shape}")

    # 5) 如果单位是 K，则转为 °C
    units = getattr(v, "units", "").lower()
    if "kelvin" in units or units == "k":
        sst = sst - 273.15
        print("单位为 Kelvin，已转换为 Celsius。")

    # 6) 保证 time 升序
    if time.ndim == 1:
        t_order = np.argsort(time)
        if not np.all(t_order == np.arange(time.size)):
            time = time[t_order]
            sst = np.take(sst, t_order, axis=t_ax)
            print("已将 time 调整为升序。")

    # 7) 保证 lat 升序（-90 -> 90）
    if lat.ndim == 1 and lat.size > 1 and lat[0] > lat[-1]:
        lat = lat[::-1]
        sst = np.flip(sst, axis=la_ax)
        print("已将 lat 由北到南改为南到北（升序）。")

    # 8) 将 lon 统一为 [-180, 180) 并升序
    #    若原始是 0..360，把它映射到 -180..180
    lon2 = lon.copy()
    if lon2.max() > 180.0:  # 说明大概率是 0..360
        lon2 = ((lon2 + 180.0) % 360.0) - 180.0
        print("已将经度从 0..360 转为 [-180,180)。")

    # 升序重排 lon（以及 sst 对应轴）
    lo_order = np.argsort(lon2)
    if not np.all(lo_order == np.arange(lon2.size)):
        lon2 = lon2[lo_order]
        sst  = np.take(sst, lo_order, axis=lo_ax)
        print("已按经度升序重排经度与数据。")
    lon = lon2

    # 9) 强制 dtype 与 缺测
    sst = to_float32(sst)
    lat = to_float32(lat)
    lon = to_float32(lon)
    time = to_float32(time)

    # 10) 最终确认形状为 (time, lat, lon)
    #     如果当前轴顺序不是 (t_ax, la_ax, lo_ax) == (0,1,2)，就转置到这个顺序
    if (t_ax, la_ax, lo_ax) != (0, 1, 2):
        perm = np.argsort([t_ax, la_ax, lo_ax])
        # 这里需要把 sst 的轴按 (time, lat, lon) 的顺序调过来
        sst = np.transpose(sst, axes=(t_ax, la_ax, lo_ax))
        print("已将数据重排为 (time, lat, lon)。")

# 11) 保存为 .npy（你要求 .npy，我就分别保存）
np.save(out_data, sst)
np.save(out_lat, lat)
np.save(out_lon, lon)
np.save(out_time, time)

print(f"Saved {out_data} with shape {sst.shape}, dtype={sst.dtype}")
print(f"Saved {out_lat}, {out_lon}, {out_time}")
print("Standardization done.")
