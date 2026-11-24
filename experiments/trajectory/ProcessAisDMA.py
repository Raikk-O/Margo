import json
import os
import pandas as pd
import numpy as np
import math

INPUT_FILE = r"D:\SBFA\datasets\DMA AIS\aisdk-2026-08-26.csv"

OUTPUT_DIR = r"./AIS/DMA/aisdk-2026-08-26"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEGMENTS_FILE = os.path.join(OUTPUT_DIR, "segments.csv")
WINDOWS_FILE = os.path.join(OUTPUT_DIR, "windows.csv")
NORM_FILE = os.path.join(OUTPUT_DIR, "normalization.json")

RESAMPLE_FREQ = "10min"  
MIN_POINTS = 20  
MIN_DURATION_HOURS = 4  
SPLIT_GAP_MINUTES = 120  
MAX_SEGMENT_HOURS = 20  
WINDOW_SIZE = 42  
WINDOW_STEP = 21  

MOORED_ANCHORED_STATUS = {
    "Moored",
    "Anchored"
}

FEATURE_COLS = ["Latitude", "Longitude", "SOG", "COG", "Heading"]

TREND_WINDOW = 5  
MAX_SOG_CHANGE = 2.0  
MAX_COG_CHANGE = 5.0  
WEIGHT_SOG = 0.5
WEIGHT_COG = 0.5

SURROUNDING_RADIUS_KM = 5.0


def load_data(path: str):
    
    print(f"正在读取数据: {path}")
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["# Timestamp"], dayfirst=True)

    required = ["MMSI", "Timestamp", "Navigational status"] + FEATURE_COLS
    df = df[required].copy()

    df["Heading"] = df["Heading"].where((df["Heading"] >= 0) & (df["Heading"] <= 360), np.nan)

    df = df.dropna(subset=["MMSI"])
    df = df.sort_values(["MMSI", "Timestamp"]).reset_index(drop=True)

    raw_df = df.copy()
    print(f"原始记录数: {len(raw_df)}")
    return raw_df, df


def filter_moored_anchored(df: pd.DataFrame) -> pd.DataFrame:
    status_col = "Navigational status"
    before = len(df)
    mask = df[status_col].isin(MOORED_ANCHORED_STATUS)
    df = df[~mask].copy()
    print(f"过滤停泊/锚泊后记录数: {len(df)} (移除 {before - len(df)})")
    return df


def filter_high_speed(df: pd.DataFrame, max_sog: float = 30.0) -> pd.DataFrame:
    before = len(df)
    df = df[df["SOG"] <= max_sog].copy()
    print(f"过滤 SOG > {max_sog} 节后记录数: {len(df)} (移除 {before - len(df)})")
    return df


def split_segments(df: pd.DataFrame) -> list:
    segments = []
    for mmsi, group in df.groupby("MMSI"):
        group = group.sort_values("Timestamp").copy()
        time_diff = group["Timestamp"].diff().dt.total_seconds() / 60.0
        split_idx = time_diff[time_diff > SPLIT_GAP_MINUTES].index

        split_positions = [0]
        for idx in split_idx:
            pos = group.index.get_loc(idx)
            split_positions.append(pos)
        split_positions.append(len(group))

        for i in range(len(split_positions) - 1):
            seg = group.iloc[split_positions[i]:split_positions[i + 1]].copy()
            segments.append(seg)
    print(f"分段后航段数: {len(segments)}")
    return segments


def filter_short_segments(segments: list) -> list:
    valid = []
    for seg in segments:
        if len(seg) < MIN_POINTS:
            continue
        duration_hours = (seg["Timestamp"].max() - seg["Timestamp"].min()).total_seconds() / 3600.0
        if duration_hours < MIN_DURATION_HOURS:
            continue
        valid.append(seg)
    print(f"保留航段数: {len(valid)} (至少 {MIN_POINTS} 点且持续 >= {MIN_DURATION_HOURS} 小时)")
    return valid


def resample_segment(seg: pd.DataFrame) -> pd.DataFrame:
    mmsi = seg['MMSI'].iloc[0]
    seg = seg.set_index("Timestamp").sort_index()

    seg['COG_sin'] = np.sin(np.deg2rad(seg['COG']))
    seg['COG_cos'] = np.cos(np.deg2rad(seg['COG']))

    interp_cols = ["Latitude", "Longitude", "SOG", "Heading", "COG_sin", "COG_cos"]
    resampled = seg[interp_cols].resample(RESAMPLE_FREQ).mean()

    for col in interp_cols:
        resampled[col] = resampled[col].interpolate(method="linear", limit_area="inside")

    resampled = resampled.ffill().bfill().dropna()

    resampled['COG'] = np.rad2deg(np.arctan2(resampled['COG_sin'], resampled['COG_cos'])) % 360
    resampled = resampled.drop(columns=['COG_sin', 'COG_cos'])

    resampled = resampled.reset_index()
    resampled['MMSI'] = mmsi
    return resampled


def resample_segments(segments: list) -> list:
    resampled = []
    for seg in segments:
        rs = resample_segment(seg)
        if len(rs) >= MIN_POINTS:
            resampled.append(rs)
    print(f"重采样后保留航段数: {len(resampled)}")
    return resampled


def split_long_segments(segments: list, max_hours: float = MAX_SEGMENT_HOURS) -> list:
    new_segments = []
    for seg in segments:
        duration_hours = (seg["Timestamp"].max() - seg["Timestamp"].min()).total_seconds() / 3600.0
        if duration_hours <= max_hours:
            new_segments.append(seg)
            continue

        sub_duration = pd.Timedelta(hours=max_hours / 2)
        start_time = seg["Timestamp"].min()
        end_time = seg["Timestamp"].max()

        while start_time < end_time:
            sub_end = min(start_time + sub_duration, end_time)
            sub_seg = seg[(seg["Timestamp"] >= start_time) & (seg["Timestamp"] < sub_end)].copy()
            if len(sub_seg) >= MIN_POINTS:
                new_segments.append(sub_seg)
            start_time = sub_end

    print(f"长航段分割后航段数: {len(new_segments)}")
    return new_segments


def add_trend_value(segments: list) -> list:
    for seg in segments:
        seg['trend_value'] = np.nan
        for mmsi, group in seg.groupby('MMSI'):
            group = group.sort_values('Timestamp')

            delta_sog = group['SOG'].diff().abs()
            delta_cog = ((group['COG'].diff() + 180) % 360 - 180).abs()

            smooth_sog = delta_sog.rolling(TREND_WINDOW, min_periods=1).mean()
            smooth_cog = delta_cog.rolling(TREND_WINDOW, min_periods=1).mean()

            norm_sog = smooth_sog.clip(0, MAX_SOG_CHANGE) / MAX_SOG_CHANGE
            norm_cog = smooth_cog.clip(0, MAX_COG_CHANGE) / MAX_COG_CHANGE

            trend = WEIGHT_SOG * norm_sog + WEIGHT_COG * norm_cog
            seg.loc[group.index, 'trend_value'] = trend

        seg['trend_value'] = seg['trend_value'].bfill().fillna(0)

    print("已为所有航段添加 trend_value 列")
    return segments


def sliding_window(segments: list) -> list:
    windows = []
    for seg_id, seg in enumerate(segments):
        n = len(seg)
        if n < WINDOW_SIZE:
            continue
        start = 0
        while start + WINDOW_SIZE <= n:
            window = seg.iloc[start:start + WINDOW_SIZE].copy()
            window["segment_id"] = seg_id
            window["window_id"] = len(windows)
            windows.append(window)
            start += WINDOW_STEP
    print(f"滑动窗口数量: {len(windows)}")
    return windows


def min_max_normalize(windows: list):
    print("正在计算全局归一化参数...")
    all_data = pd.concat(windows, ignore_index=True)
    stats = {}
    for col in FEATURE_COLS:
        cmin = float(all_data[col].min())
        cmax = float(all_data[col].max())
        stats[col] = {"min": cmin, "max": cmax}

    print("正在应用归一化...")
    for w in windows:
        for col in FEATURE_COLS:
            cmin, cmax = stats[col]["min"], stats[col]["max"]
            if cmax > cmin:
                w[f"{col}_norm"] = (w[col] - cmin) / (cmax - cmin)
            else:
                w[f"{col}_norm"] = 0.0
    return windows, stats


def filter_coastline(segments: list) -> list:
    print("⚠️ 警告：近岸过滤（<1 nmi）未实现。缺少海岸线矢量数据，该步骤已跳过。请在论文中说明。")
    return segments


def haversine_vectorized(lat1, lon1, lat2_array, lon2_array):
    
    R = 6371.0
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2 = np.radians(lat2_array)
    lon2 = np.radians(lon2_array)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def get_surrounding_info(segment_group, global_df, radius_km):
    
    if global_df is None or global_df.empty:
        return "No surrounding vessels information available."

    start_time = segment_group['Timestamp'].min()
    end_time = segment_group['Timestamp'].max()
    own_mmsi = segment_group['MMSI'].iloc[0]
    center_lat = segment_group['Latitude'].mean()
    center_lon = segment_group['Longitude'].mean()

    mask_time = (global_df['Timestamp'] >= start_time) & (global_df['Timestamp'] <= end_time)
    candidates = global_df[mask_time]
    if candidates.empty:
        return "No other vessels in the same time window."

    lat_delta = radius_km / 111.0
    cos_lat = math.cos(math.radians(center_lat))
    lon_delta = radius_km / (111.0 * cos_lat) if cos_lat > 1e-6 else 360.0

    mask_box = (
            (candidates['Latitude'] >= center_lat - lat_delta) &
            (candidates['Latitude'] <= center_lat + lat_delta) &
            (candidates['Longitude'] >= center_lon - lon_delta) &
            (candidates['Longitude'] <= center_lon + lon_delta)
    )
    candidates = candidates[mask_box]

    candidates = candidates[candidates['MMSI'] != own_mmsi]
    if candidates.empty:
        return f"No other vessels within {radius_km} km during this period."

    distances = haversine_vectorized(
        center_lat, center_lon,
        candidates['Latitude'].values,
        candidates['Longitude'].values
    )

    valid_mask = distances <= radius_km
    valid_candidates = candidates[valid_mask]
    valid_distances = distances[valid_mask]

    if len(valid_candidates) == 0:
        return f"No other vessels within {radius_km} km during this period."

    valid_candidates = valid_candidates.copy()
    valid_candidates['dist'] = valid_distances
    grouped = valid_candidates.groupby('MMSI')['dist'].min()

    num = len(grouped)
    closest_mmsi = grouped.idxmin()
    closest_dist = grouped.min()

    return (f"There were {num} other vessel(s) within {radius_km} km. "
            f"The closest was {closest_dist:.2f} km away (MMSI: {closest_mmsi}).")


def add_surrounding_info(windows, raw_df, radius_km):
    
    print("正在计算周围船只信息（已向量化优化）...")
    total = len(windows)
    for i, w in enumerate(windows):
        info = get_surrounding_info(w, raw_df, radius_km)
        w['surrounding_info'] = info
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"  进度: {i + 1}/{total}")
    print("周围信息添加完成")
    return windows


def save_results(segments, windows, stats):
    if segments:
        pd.concat(segments, ignore_index=True).to_csv(SEGMENTS_FILE, index=False)
        print(f"已保存航段文件: {SEGMENTS_FILE}")
    if windows:
        pd.concat(windows, ignore_index=True).to_csv(WINDOWS_FILE, index=False)
        print(f"已保存窗口文件: {WINDOWS_FILE}")
    with open(NORM_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"已保存归一化参数: {NORM_FILE}")


def main():
    raw_df, df = load_data(INPUT_FILE)

    df = filter_moored_anchored(df)
    df = filter_high_speed(df)

    segments = split_segments(df)
    segments = filter_short_segments(segments)

    segments = filter_coastline(segments)

    segments = resample_segments(segments)

    segments = split_long_segments(segments)

    segments = add_trend_value(segments)

    windows = sliding_window(segments)

    windows = add_surrounding_info(windows, raw_df, SURROUNDING_RADIUS_KM)

    windows, stats = min_max_normalize(windows)

    save_results(segments, windows, stats)


if __name__ == "__main__":
    main()