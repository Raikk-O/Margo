
import argparse
import json
import os

import numpy as np
import pandas as pd

TOTAL_LEN = 42           
HIST_LEN = 18            
MAX_SOG = 30.0           

FEATURE_COLS = ["dLat", "dLon", "SOG", "COG"]

DATA_DIR = r"D:\SBFA\datasets\AmericanAIS"  
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "AmericanAis")

MOORED_ANCHORED_STATUS = {1, 5, 6}

USECOLS = ["mmsi", "base_date_time", "longitude", "latitude", "sog", "cog", "heading", "status"]

PORTS = {
    "houston": (29.0, 30.1, -95.6, -94.2),   
    "losangeles": (33.4, 34.0, -118.6, -117.8),
    "newyork": (40.2, 41.0, -74.5, -73.5),
    "neworleans": (28.7, 30.3, -90.6, -89.1),
}


def load_day(path):
    df = pd.read_csv(path, usecols=USECOLS)
    df = df.rename(columns={"mmsi": "vid", "base_date_time": "t",
                            "longitude": "lon", "latitude": "lat"})
    df["t"] = pd.to_datetime(df["t"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    for c in ["lon", "lat", "sog", "cog", "heading", "status"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["t", "lon", "lat", "sog", "cog"])
    return df


def filter_bbox(df, bbox):
    
    lat_min, lat_max, lon_min, lon_max = bbox
    before = len(df)
    df = df[(df["lat"] >= lat_min) & (df["lat"] <= lat_max) &
            (df["lon"] >= lon_min) & (df["lon"] <= lon_max)]
    print("  [框定] %s -> %d -> %d (移除 %.1f%%)"
          % (bbox, before, len(df), 100 * (1 - len(df) / max(before, 1))))
    return df


def filter_invalid(df, min_speed, max_sog):
    before = len(df)
    df = df[(df["sog"] >= 0) & (df["sog"] <= max_sog)]
    df = df[(df["cog"] >= 0) & (df["cog"] < 360)]
    df = df[df["status"].isna() | (~df["status"].isin(MOORED_ANCHORED_STATUS))]
    df = df[df["sog"] >= min_speed]
    df = df.drop_duplicates(subset=["vid", "t"])
    df = df.sort_values(["vid", "t"]).reset_index(drop=True)
    print("  [清洗] %d -> %d (移除 %.1f%%)" % (before, len(df), 100 * (1 - len(df) / max(before, 1))))
    return df


def _group_runs(seg_no):
    seg_no = np.asarray(seg_no)
    starts = np.r_[0, np.flatnonzero(seg_no[1:] != seg_no[:-1]) + 1]
    ends = np.r_[starts[1:], len(seg_no)]
    for s, e in zip(starts, ends):
        yield seg_no[s], np.arange(s, e)


def segment_and_window(df, rate, gap, min_len, stride, seg_start=0):
    gap_td = pd.Timedelta(gap)
    windows = []
    n_seg_kept = 0
    n_seg_total = 0

    for vid, g in df.groupby("vid", sort=False):
        g = g.set_index("t").sort_index()
        if len(g) < 2:
            continue
        is_break = (g.index.to_series().diff() > gap_td).to_numpy(dtype=bool)
        seg_no = np.cumsum(is_break)

        for _, positions in _group_runs(seg_no):
            gg = g.iloc[positions]
            if len(gg) < 2:
                continue
            n_seg_total += 1
            tmp = pd.DataFrame({
                "lat": gg["lat"].values,
                "lon": gg["lon"].values,
                "sog": gg["sog"].values,
                "cos": np.cos(np.deg2rad(gg["cog"].values)),
                "sin": np.sin(np.deg2rad(gg["cog"].values)),
            }, index=gg.index)
            rs = tmp.resample(rate).mean()
            rs = rs.interpolate(method="linear", limit_direction="both")
            rs["cog"] = np.rad2deg(np.arctan2(rs["sin"], rs["cos"])) % 360
            rs = rs.dropna(subset=["lat", "lon", "sog", "cog"])

            if len(rs) < min_len:
                continue
            n_seg_kept += 1
            sid = str(seg_start + n_seg_kept - 1)
            for start in range(0, len(rs) - TOTAL_LEN + 1, stride):
                w = rs.iloc[start:start + TOTAL_LEN]
                lat_arr = w["lat"].to_numpy(dtype=np.float64)
                lon_arr = w["lon"].to_numpy(dtype=np.float64)
                anchor_lat = float(lat_arr[HIST_LEN - 1])
                anchor_lon = float(lon_arr[HIST_LEN - 1])
                windows.append(pd.DataFrame({
                    "Latitude": lat_arr,
                    "Longitude": lon_arr,
                    "SOG": w["sog"].to_numpy(dtype=np.float64),
                    "COG": w["cog"].to_numpy(dtype=np.float64),
                    "anchor_lat": anchor_lat,
                    "anchor_lon": anchor_lon,
                    "dLat": lat_arr - anchor_lat,
                    "dLon": lon_arr - anchor_lon,
                    "segment_id": sid,
                }))

    print("  [切段] 总段数 %d, 保留(>=%d点) %d, 滑窗样本 %d"
          % (n_seg_total, min_len, n_seg_kept, len(windows)))
    return windows, n_seg_kept


def main():
    ap = argparse.ArgumentParser(description="AmericanAIS 数据清洗")
    ap.add_argument("--rate", default="120s", help="重采样间隔")
    ap.add_argument("--gap", default="1800s", help="同一船时间间隙超过该值切段")
    ap.add_argument("--min-speed", type=float, default=0.5, help="低于该速度(节)判为停船")
    ap.add_argument("--max-sog", type=float, default=MAX_SOG)
    ap.add_argument("--min-len", type=int, default=TOTAL_LEN)
    ap.add_argument("--stride", type=int, default=TOTAL_LEN)
    ap.add_argument("--output", default=os.path.join(OUTPUT_DIR, "windows_america.csv"))
    ap.add_argument("--files", default=None, help="逗号分隔的文件名; 默认全部 .csv")
    ap.add_argument("--bbox", default=None,
                    help="地理框定: 港口名(houston)或 lat_min,lat_max,lon_min,lon_max; 默认不框定")
    args = ap.parse_args()

    bbox = None
    if args.bbox:
        if args.bbox in PORTS:
            bbox = PORTS[args.bbox]
        else:
            parts = [float(x) for x in args.bbox.split(",") if x.strip()]
            if len(parts) != 4:
                raise SystemExit("--bbox 需为港口名或 4 个坐标 lat_min,lat_max,lon_min,lon_max")
            bbox = tuple(parts)

    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    if args.files:
        files = [f.strip() for f in args.files.split(",") if f.strip()]

    all_windows = []
    seg_offset = 0
    for fn in files:
        path = os.path.join(DATA_DIR, fn)
        if not os.path.exists(path):
            print("[跳过] %s 不存在" % path)
            continue
        print("\n===== %s =====" % fn)
        df = load_day(path)
        print("  原始 %d 行, %d 艘船" % (len(df), df["vid"].nunique()))
        if bbox is not None:
            df = filter_bbox(df, bbox)
        df = filter_invalid(df, args.min_speed, args.max_sog)
        windows, n_seg = segment_and_window(df, args.rate, args.gap,
                                            args.min_len, args.stride, seg_offset)
        all_windows.extend(windows)
        seg_offset += n_seg
        print("  累计样本: %d" % len(all_windows))

    if not all_windows:
        raise SystemExit("没有生成任何窗口: 请放宽 --gap / --min-speed")

    out = pd.concat(all_windows, ignore_index=True)
    n_win = len(out) // TOTAL_LEN
    out = out.iloc[:n_win * TOTAL_LEN]

    stats = {}
    for col in FEATURE_COLS:
        cmin = float(out[col].min())
        cmax = float(out[col].max())
        stats[col] = {"min": cmin, "max": cmax}
        out[f"{col}_norm"] = (out[col] - cmin) / (cmax - cmin) if cmax > cmin else 0.0
    norm_json = os.path.splitext(os.path.abspath(args.output))[0] + "_norm.json"
    with open(norm_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("  归一化参数: %s" % json.dumps(stats))
    print("  归一化参数已保存: %s" % norm_json)

    block = out["segment_id"].values.reshape(n_win, TOTAL_LEN)
    const = (block == block[:, :1]).all(axis=1).mean()
    seg_rows = out.groupby("segment_id").size()
    print("\n[质量报告] 输出 %d 行 = %d 样本 x %d 步" % (len(out), n_win, TOTAL_LEN))
    print("  SOG==0 占比: %.2f%%   SOG<0.5 占比: %.2f%%   平均 SOG: %.3f kn"
          % (100 * (out["SOG"] == 0).mean(), 100 * (out["SOG"] < 0.5).mean(), out["SOG"].mean()))
    print("  42块内 segment_id 恒定占比: %.1f%%" % (100 * const))
    print("  segment 行数为 42 倍数占比: %.1f%%" % (100 * (seg_rows % TOTAL_LEN == 0).mean()))
    print("  段数 %d" % out["segment_id"].nunique())

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print("  已保存: %s" % os.path.abspath(args.output))


if __name__ == "__main__":
    main()
