import json
import os
import pandas as pd
import numpy as np
import math

# -----------------------------
# 1. 配置参数（请根据实际路径修改）
# -----------------------------
INPUT_FILE = r"D:\SBFA\datasets\DMA AIS\aisdk-2026-08-26.csv"

# 确保输出目录存在
OUTPUT_DIR = r"./AIS/DMA/aisdk-2026-08-26"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEGMENTS_FILE = os.path.join(OUTPUT_DIR, "segments.csv")
WINDOWS_FILE = os.path.join(OUTPUT_DIR, "windows.csv")
NORM_FILE = os.path.join(OUTPUT_DIR, "normalization.json")

# DMA 专用参数（严格对齐论文 A.2.2）
RESAMPLE_FREQ = "10min"  # 10 分钟重采样
MIN_POINTS = 20  # 论文: at least 20 AIS messages
MIN_DURATION_HOURS = 4  # 论文: minimum of four hours
SPLIT_GAP_MINUTES = 120  # 间隔 > 2 小时分段
MAX_SEGMENT_HOURS = 20  # 长航段超过 20 小时再分割
WINDOW_SIZE = 42  # 滑动窗口长度（模型输入序列长度）
WINDOW_STEP = 21  # 窗口步长（重叠 21 点）

MOORED_ANCHORED_STATUS = {
    "Moored",
    "Anchored"
}

# 特征列（论文只使用位置、SOG、COG，但 Heading 也可保留）
FEATURE_COLS = ["Latitude", "Longitude", "SOG", "COG", "Heading"]

# 趋势值计算参数（针对 10 分钟采样）
TREND_WINDOW = 5  # 滑动平均窗口（点数）
MAX_SOG_CHANGE = 2.0  # 节/10分钟（10分钟内速度变化上限）
MAX_COG_CHANGE = 5.0  # 度/10分钟
WEIGHT_SOG = 0.5
WEIGHT_COG = 0.5

# 周围船只搜索半径（公里）
SURROUNDING_RADIUS_KM = 5.0


# -----------------------------
# 2. 数据读取
# -----------------------------
def load_data(path: str):
    """
    读取原始数据，返回：
        raw_df: 未做任何过滤的原始数据（用于周围信息查询）
        df: 初步清洗后的副本（后续处理）
    """
    print(f"正在读取数据: {path}")
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["# Timestamp"], dayfirst=True)

    required = ["MMSI", "Timestamp", "Navigational status"] + FEATURE_COLS
    df = df[required].copy()

    # ✅ 优化：清洗 Heading（511 及其他 <0 或 >360 的异常值均置为 NaN）
    df["Heading"] = df["Heading"].where((df["Heading"] >= 0) & (df["Heading"] <= 360), np.nan)

    df = df.dropna(subset=["MMSI"])
    df = df.sort_values(["MMSI", "Timestamp"]).reset_index(drop=True)

    # 保存原始数据（未过滤）
    raw_df = df.copy()
    print(f"原始记录数: {len(raw_df)}")
    return raw_df, df


# -----------------------------
# 3. 过滤停泊/锚泊
# -----------------------------
def filter_moored_anchored(df: pd.DataFrame) -> pd.DataFrame:
    status_col = "Navigational status"
    before = len(df)
    mask = df[status_col].isin(MOORED_ANCHORED_STATUS)
    df = df[~mask].copy()
    print(f"过滤停泊/锚泊后记录数: {len(df)} (移除 {before - len(df)})")
    return df


# -----------------------------
# 4. 过滤 SOG > 30 节
# -----------------------------
def filter_high_speed(df: pd.DataFrame, max_sog: float = 30.0) -> pd.DataFrame:
    before = len(df)
    df = df[df["SOG"] <= max_sog].copy()
    print(f"过滤 SOG > {max_sog} 节后记录数: {len(df)} (移除 {before - len(df)})")
    return df


# -----------------------------
# 5. 按 MMSI 分段（间隔 > 2 小时）
# -----------------------------
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


# -----------------------------
# 6. 过滤短航段
# -----------------------------
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


# -----------------------------
# 7. 重采样为 10 分钟间隔（安全的 COG 插值）
# -----------------------------
def resample_segment(seg: pd.DataFrame) -> pd.DataFrame:
    mmsi = seg['MMSI'].iloc[0]
    seg = seg.set_index("Timestamp").sort_index()

    # ✅ 优化：COG 转 sin/cos 后再插值，避免 359°->1° 产生 180° 的错误中间值
    seg['COG_sin'] = np.sin(np.deg2rad(seg['COG']))
    seg['COG_cos'] = np.cos(np.deg2rad(seg['COG']))

    interp_cols = ["Latitude", "Longitude", "SOG", "Heading", "COG_sin", "COG_cos"]
    resampled = seg[interp_cols].resample(RESAMPLE_FREQ).mean()

    for col in interp_cols:
        resampled[col] = resampled[col].interpolate(method="linear", limit_area="inside")

    resampled = resampled.ffill().bfill().dropna()

    # 还原 COG
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


# -----------------------------
# 8. 长航段再分割（>20 小时）
# -----------------------------
def split_long_segments(segments: list, max_hours: float = MAX_SEGMENT_HOURS) -> list:
    new_segments = []
    for seg in segments:
        duration_hours = (seg["Timestamp"].max() - seg["Timestamp"].min()).total_seconds() / 3600.0
        if duration_hours <= max_hours:
            new_segments.append(seg)
            continue

        # ✅ 优化：参数化子段时长
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


# -----------------------------
# 9. 计算趋势值
# -----------------------------
def add_trend_value(segments: list) -> list:
    for seg in segments:
        seg['trend_value'] = np.nan
        for mmsi, group in seg.groupby('MMSI'):
            group = group.sort_values('Timestamp')

            delta_sog = group['SOG'].diff().abs()
            # ✅ 优化：安全的角度差计算，处理所有边界情况
            delta_cog = ((group['COG'].diff() + 180) % 360 - 180).abs()

            smooth_sog = delta_sog.rolling(TREND_WINDOW, min_periods=1).mean()
            smooth_cog = delta_cog.rolling(TREND_WINDOW, min_periods=1).mean()

            norm_sog = smooth_sog.clip(0, MAX_SOG_CHANGE) / MAX_SOG_CHANGE
            norm_cog = smooth_cog.clip(0, MAX_COG_CHANGE) / MAX_COG_CHANGE

            trend = WEIGHT_SOG * norm_sog + WEIGHT_COG * norm_cog
            seg.loc[group.index, 'trend_value'] = trend

        # ✅ 优化：替换已弃用的 fillna(method='bfill')
        seg['trend_value'] = seg['trend_value'].bfill().fillna(0)

    print("已为所有航段添加 trend_value 列")
    return segments


# -----------------------------
# 10. 滑动窗口切分
# -----------------------------
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


# -----------------------------
# 11. Min-Max 归一化
# -----------------------------
def min_max_normalize(windows: list):
    # ✅ 优化：合并后一次性计算全局统计量，避免逐窗口循环
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


# -----------------------------
# 12. 近岸过滤（占位）
# -----------------------------
def filter_coastline(segments: list) -> list:
    print("⚠️ 警告：近岸过滤（<1 nmi）未实现。缺少海岸线矢量数据，该步骤已跳过。请在论文中说明。")
    return segments


# -----------------------------
# 13. 周围船只信息计算函数 (向量化优化)
# -----------------------------
def haversine_vectorized(lat1, lon1, lat2_array, lon2_array):
    """向量化 Haversine 计算"""
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
    """查询指定窗口周围的其他船只（优化版）"""
    if global_df is None or global_df.empty:
        return "No surrounding vessels information available."

    start_time = segment_group['Timestamp'].min()
    end_time = segment_group['Timestamp'].max()
    own_mmsi = segment_group['MMSI'].iloc[0]
    center_lat = segment_group['Latitude'].mean()
    center_lon = segment_group['Longitude'].mean()

    # 1. 时间过滤
    mask_time = (global_df['Timestamp'] >= start_time) & (global_df['Timestamp'] <= end_time)
    candidates = global_df[mask_time]
    if candidates.empty:
        return "No other vessels in the same time window."

    # 2. 空间粗过滤（矩形框）
    lat_delta = radius_km / 111.0
    # 防止高纬度地区 cos 接近 0 导致除零错误
    cos_lat = math.cos(math.radians(center_lat))
    lon_delta = radius_km / (111.0 * cos_lat) if cos_lat > 1e-6 else 360.0

    mask_box = (
            (candidates['Latitude'] >= center_lat - lat_delta) &
            (candidates['Latitude'] <= center_lat + lat_delta) &
            (candidates['Longitude'] >= center_lon - lon_delta) &
            (candidates['Longitude'] <= center_lon + lon_delta)
    )
    candidates = candidates[mask_box]

    # 排除自身
    candidates = candidates[candidates['MMSI'] != own_mmsi]
    if candidates.empty:
        return f"No other vessels within {radius_km} km during this period."

    # 3. 向量化精确距离计算
    distances = haversine_vectorized(
        center_lat, center_lon,
        candidates['Latitude'].values,
        candidates['Longitude'].values
    )

    # 过滤出半径内的船只
    valid_mask = distances <= radius_km
    valid_candidates = candidates[valid_mask]
    valid_distances = distances[valid_mask]

    if len(valid_candidates) == 0:
        return f"No other vessels within {radius_km} km during this period."

    # 统计去重（取每艘船最近的距离）
    valid_candidates = valid_candidates.copy()
    valid_candidates['dist'] = valid_distances
    grouped = valid_candidates.groupby('MMSI')['dist'].min()

    num = len(grouped)
    closest_mmsi = grouped.idxmin()
    closest_dist = grouped.min()

    return (f"There were {num} other vessel(s) within {radius_km} km. "
            f"The closest was {closest_dist:.2f} km away (MMSI: {closest_mmsi}).")


def add_surrounding_info(windows, raw_df, radius_km):
    """为每个窗口添加 surrounding_info 列"""
    print("正在计算周围船只信息（已向量化优化）...")
    total = len(windows)
    for i, w in enumerate(windows):
        info = get_surrounding_info(w, raw_df, radius_km)
        w['surrounding_info'] = info
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"  进度: {i + 1}/{total}")
    print("周围信息添加完成")
    return windows


# -----------------------------
# 14. 保存结果
# -----------------------------
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


# -----------------------------
# 15. 主流程
# -----------------------------
def main():
    # 1. 加载数据
    raw_df, df = load_data(INPUT_FILE)

    # 2. 过滤（停泊/锚泊、超速）
    df = filter_moored_anchored(df)
    df = filter_high_speed(df)

    # 3. 分段
    segments = split_segments(df)
    segments = filter_short_segments(segments)

    # 4. 近岸过滤（占位）
    segments = filter_coastline(segments)

    # 5. 重采样
    segments = resample_segments(segments)

    # 6. 长航段再分割
    segments = split_long_segments(segments)

    # 7. 计算趋势值
    segments = add_trend_value(segments)

    # 8. 滑动窗口
    windows = sliding_window(segments)

    # 9. 计算周围船只信息（使用原始数据）
    windows = add_surrounding_info(windows, raw_df, SURROUNDING_RADIUS_KM)

    # 10. 归一化
    windows, stats = min_max_normalize(windows)

    # 11. 保存结果
    save_results(segments, windows, stats)


if __name__ == "__main__":
    main()