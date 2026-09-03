#!/usr/bin/env python3
"""
analyse_attention.py
--------------------
Parses Tobii Studio binary recordings and computes per-AOI-category
attention metrics (dwell time, gaze hit count) across all participants.

Data chain:
  ElementPresentation.data  -> Stimulus Key (Windows GUID)
  Stimulus/<key>.exp        -> <MediaId>
  SQLite Media.MediaKey     -> Media.Id + image dimensions
  SQLite Aoi.Media_Id       -> polygon vertices (image-pixel coords)
  SQLite AoiTagAoi + AoiTag -> category name (e.g. "乔木", "水体")

Outputs (written next to this script):
  attention_by_category.csv   aggregate across all recordings & stimuli
  attention_by_stimulus.csv   per stimulus, per category
  attention_by_recording.csv  per recording, per stimulus, per category
"""

import struct
import sqlite3
import sys
import io
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent #gets the directory of the current script, which is assumed to be the project root
PROJECT_ROOT = _HERE / "Data" # builds path to data folder
RECORDINGS_DIR = PROJECT_ROOT / "Recordings" 
STIMULUS_DIR   = PROJECT_ROOT / "Stimulus"
DB_PATH        = PROJECT_ROOT / "AnalysisData" / "AnalysisData.db"
OUTPUT_DIR     = _HERE

# ---------------------------------------------------------------------------
# Binary format constants
# ---------------------------------------------------------------------------
ET_RECORD_SIZE = 124   # bytes per Eyetracker.data sample
EP_RECORD_SIZE = 48    # bytes per ElementPresentation.data record

# Within the 24-float block (starting at byte 20 of each ET record):
#   float[ 7] = left  gaze X normalized (UCS 0-1)
#   float[ 8] = left  gaze Y normalized
#   float[19] = right gaze X normalized
#   float[20] = right gaze Y normalized
# Slots in a 31-slot uint32/float32 view of 124 bytes:
#   slot 0  = type        (bytes  0-3)
#   slot 1-2 = dev_ts     (bytes  4-11)
#   slot 3-4 = sys_ts     (bytes 12-19)
#   slots 5-28 = 24 floats (bytes 20-115)  -> slot 5+j for float[j]
#   slot 29 = validity_left  (bytes 116-119)
#   slot 30 = validity_right (bytes 120-123)
_FLOAT_SLOT_OFFSET = 5
_SLOT_GAZE_LX = _FLOAT_SLOT_OFFSET + 7
_SLOT_GAZE_LY = _FLOAT_SLOT_OFFSET + 8
_SLOT_GAZE_RX = _FLOAT_SLOT_OFFSET + 19
_SLOT_GAZE_RY = _FLOAT_SLOT_OFFSET + 20
_SLOT_VAL_L   = 29
_SLOT_VAL_R   = 30

# Gaze filter: reject samples whose normalized coords are wildly off-screen
_GAZE_MIN = -0.5
_GAZE_MAX =  1.5

# ---------------------------------------------------------------------------
# Helper: Windows GUID bytes -> lowercase UUID string
# ---------------------------------------------------------------------------
def _bytes_to_guid(b: bytes) -> str:
    p1 = struct.unpack_from("<I", b, 0)[0]
    p2 = struct.unpack_from("<H", b, 4)[0]
    p3 = struct.unpack_from("<H", b, 6)[0]
    p4 = b[8:16]
    return f"{p1:08x}-{p2:04x}-{p3:04x}-{p4[:2].hex()}-{p4[2:].hex()}"


# ---------------------------------------------------------------------------
# Step 1: Build stimulus-key -> media_id lookup from Stimulus/*.exp files
# ---------------------------------------------------------------------------
def build_stimulus_lookup(stimulus_dir: Path) -> dict:
    """
    Returns: {stimulus_key_lower: (media_id_lower, stimulus_name)}
    """
    lookup = {}
    for f in stimulus_dir.glob("*.exp"):
        try:
            root = ET.parse(f).getroot()
            key = (root.findtext("Key") or "").strip().lower()
            if not key:
                continue
            # MediaId lives inside the Value element
            media_id = (
                root.findtext(".//{*}MediaId")
                or root.findtext(".//MediaId")
                or ""
            ).strip().lower()
            name = (
                root.findtext(".//{*}Name")
                or root.findtext(".//Name")
                or f.stem
            ).strip()
            lookup[key] = (media_id, name)
        except Exception:
            pass
    return lookup


# ---------------------------------------------------------------------------
# Step 2: Load database: media info + AOI polygons + category tags
# ---------------------------------------------------------------------------
def load_database(db_path: Path):
    """
    Returns:
      media_by_mkey : {media_key_lower: {id, name, width, height}}
      aoi_by_media  : {media_db_id: [ {category, mpl_path} ]}
    """
    conn = sqlite3.connect(str(db_path))

    # Media
    cur = conn.execute("SELECT Id, MediaKey, Name, Width, Height FROM Media")
    media_by_mkey = {}
    for mid, mkey, name, w, h in cur.fetchall():
        media_by_mkey[mkey.lower()] = {
            "id": mid, "name": name, "width": float(w), "height": float(h)
        }

    # AOI tags
    cur = conn.execute("SELECT Id, Name FROM AoiTag")
    tag_name = {row[0]: row[1] for row in cur.fetchall()}

    # AOI -> tag
    cur = conn.execute("SELECT AoiTag_Id, Aoi_Id FROM AoiTagAoi")
    aoi_tag = {aoi_id: tag_name.get(tag_id, "Unknown")
               for tag_id, aoi_id in cur.fetchall()}

    # AOIs with polygon vertices (XML inside KeyFrameData column)
    cur = conn.execute("SELECT Id, KeyFrameData, Media_Id FROM Aoi")
    aoi_by_media: dict = {}
    for aoi_id, kf_xml, media_db_id in cur.fetchall():
        verts = _parse_aoi_vertices(kf_xml)
        if len(verts) < 3:
            continue
        category = aoi_tag.get(aoi_id, "Unknown")
        aoi_by_media.setdefault(media_db_id, []).append({
            "category": category,
            "mpl_path": MplPath(np.array(verts, dtype=np.float64)),
        })

    conn.close()
    return media_by_mkey, aoi_by_media


def _parse_aoi_vertices(kf_xml: str) -> list:
    try:
        root = ET.fromstring(kf_xml)
        for kf in root.findall("keyframe"):
            if kf.get("isActive", "True") != "False":
                return [
                    (float(v.get("x")), float(v.get("y")))
                    for v in kf.findall("vertex")
                ]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Step 3: Parse Eyetracker.data
# ---------------------------------------------------------------------------
def parse_eyetracker(path: Path) -> pd.DataFrame:
    """
    Returns DataFrame: timestamp_us (int64), gaze_x (float32), gaze_y (float32)
    Only valid, on-screen gaze samples are included.
    gaze_x / gaze_y are in normalized screen coords (0=left/top, 1=right/bottom).
    """
    raw = path.read_bytes()
    n_records = len(raw) // ET_RECORD_SIZE
    if n_records == 0:
        return pd.DataFrame(columns=["timestamp_us", "gaze_x", "gaze_y"])

    # View as 31 uint32 slots (124 / 4 = 31)
    arr = np.frombuffer(raw[: n_records * ET_RECORD_SIZE], dtype="<u4").reshape(
        n_records, 31
    )
    float_arr = arr.view("<f4").reshape(n_records, 31)

    # Timestamps: slots 1 (low 32 bits) and 2 (high 32 bits) of device_timestamp
    ts_lo = arr[:, 1].astype(np.uint64)
    ts_hi = arr[:, 2].astype(np.uint64)
    timestamps = (ts_hi << 32) | ts_lo

    gaze_lx = float_arr[:, _SLOT_GAZE_LX]
    gaze_ly = float_arr[:, _SLOT_GAZE_LY]
    gaze_rx = float_arr[:, _SLOT_GAZE_RX]
    gaze_ry = float_arr[:, _SLOT_GAZE_RY]

    val_l = arr[:, _SLOT_VAL_L]   # 0 = valid
    val_r = arr[:, _SLOT_VAL_R]

    valid_l = val_l == 0
    valid_r = val_r == 0
    valid_any = valid_l | valid_r

    # Best-estimate gaze: average when both valid, else whichever is valid
    gaze_x = np.where(
        valid_l & valid_r, (gaze_lx + gaze_rx) / 2.0,
        np.where(valid_l, gaze_lx, gaze_rx)
    )
    gaze_y = np.where(
        valid_l & valid_r, (gaze_ly + gaze_ry) / 2.0,
        np.where(valid_l, gaze_ly, gaze_ry)
    )

    # Filter: must be valid AND within a reasonable screen-neighborhood
    in_range = (
        valid_any
        & np.isfinite(gaze_x) & np.isfinite(gaze_y)
        & (gaze_x >= _GAZE_MIN) & (gaze_x <= _GAZE_MAX)
        & (gaze_y >= _GAZE_MIN) & (gaze_y <= _GAZE_MAX)
    )

    return pd.DataFrame({
        "timestamp_us": timestamps[in_range].astype(np.int64),
        "gaze_x":       gaze_x[in_range].astype(np.float32),
        "gaze_y":       gaze_y[in_range].astype(np.float32),
    })


# ---------------------------------------------------------------------------
# Step 4: Parse ElementPresentation.data
# ---------------------------------------------------------------------------
def parse_element_presentation(path: Path) -> list:
    """
    Returns list of (start_us, end_us, stim_guid) for 'show' windows.
    stim_guid = Windows-GUID string of GUID1 (stimulus key).
    """
    raw = path.read_bytes()
    n = len(raw) // EP_RECORD_SIZE

    active: dict = {}    # guid1 -> start_ts
    intervals = []

    for i in range(n):
        off = i * EP_RECORD_SIZE
        ts    = struct.unpack_from("<Q", raw, off + 4)[0]
        state = struct.unpack_from("<I", raw, off + 12)[0]
        guid1 = _bytes_to_guid(raw[off + 16 : off + 32])

        if state == 0:          # show
            active[guid1] = ts
        elif state == 1:        # hide
            if guid1 in active:
                intervals.append((active.pop(guid1), ts, guid1))

    return intervals


# ---------------------------------------------------------------------------
# Step 5: Coordinate conversion (normalized screen -> image pixels)
# ---------------------------------------------------------------------------
def _compute_transform(screen_w: float, screen_h: float,
                       img_w: float, img_h: float) -> tuple:
    """
    Returns (scale, offset_x, offset_y) for FitToScreen+Center layout.
    screen_pixel = img_pixel * scale + offset
    img_pixel    = (screen_pixel - offset) / scale
    """
    scale    = min(screen_w / img_w, screen_h / img_h)
    offset_x = (screen_w - img_w * scale) / 2.0
    offset_y = (screen_h - img_h * scale) / 2.0
    return scale, offset_x, offset_y


def gaze_norm_to_img_px(gaze_x: np.ndarray, gaze_y: np.ndarray,
                         screen_w: float, screen_h: float,
                         img_w: float, img_h: float) -> tuple:
    scale, ox, oy = _compute_transform(screen_w, screen_h, img_w, img_h)
    px_x = gaze_x * screen_w
    px_y = gaze_y * screen_h
    img_x = (px_x - ox) / scale
    img_y = (px_y - oy) / scale
    return img_x, img_y


# ---------------------------------------------------------------------------
# Step 6: Hit test gaze points against AOI polygons
# ---------------------------------------------------------------------------
def hit_test(img_pts: np.ndarray, aois: list) -> np.ndarray:
    """
    img_pts : (N, 2) array of (x, y) in image pixel coords
    aois    : list of {category, mpl_path}
    Returns : (N,) array of category strings; None where no AOI hit.
    """
    n = len(img_pts)
    result = np.full(n, None, dtype=object)
    for aoi in aois:
        mask = aoi["mpl_path"].contains_points(img_pts)
        # Only assign first AOI hit per point
        assign = mask & (result == None)  # noqa: E711
        if assign.any():
            result[assign] = aoi["category"]
    return result


# ---------------------------------------------------------------------------
# Step 7: Process one recording directory
# ---------------------------------------------------------------------------
def process_recording(
    rec_dir:          Path,
    stim_lookup:      dict,
    media_by_mkey:    dict,
    aoi_by_media:     dict,
    screen_w:         float = 2560.0,
    screen_h:         float = 1600.0,
) -> pd.DataFrame | None:
    """
    Returns DataFrame with columns: stimulus, category, dwell_us, gaze_count
    or None if no usable data.
    """
    et_path = rec_dir / "Eyetracker.data"
    ep_path = rec_dir / "ElementPresentation.data"
    if not et_path.exists() or not ep_path.exists():
        return None

    try:
        gaze_df = parse_eyetracker(et_path)
    except Exception as e:
        print(f"    [WARN] eyetracker parse error: {e}")
        return None

    if gaze_df.empty:
        return None

    try:
        ep_intervals = parse_element_presentation(ep_path)
    except Exception as e:
        print(f"    [WARN] element presentation parse error: {e}")
        return None

    gaze_ts = gaze_df["timestamp_us"].values
    gaze_x  = gaze_df["gaze_x"].values
    gaze_y  = gaze_df["gaze_y"].values

    rows = []

    for start_us, end_us, stim_guid in ep_intervals:
        # Resolve stimulus -> media
        media_id_key, stim_name = stim_lookup.get(stim_guid.lower(), (None, None))
        if not media_id_key:
            continue
        media = media_by_mkey.get(media_id_key)
        if media is None:
            continue
        aois = aoi_by_media.get(media["id"])
        if not aois:
            continue   # skip stimuli without AOIs (text screens, calibration…)

        # Gaze samples within this stimulus window
        mask = (gaze_ts >= start_us) & (gaze_ts <= end_us)
        if not mask.any():
            continue

        ts_win = gaze_ts[mask]
        gx_win = gaze_x[mask].astype(np.float64)
        gy_win = gaze_y[mask].astype(np.float64)

        # Convert to image pixel coordinates
        img_x, img_y = gaze_norm_to_img_px(
            gx_win, gy_win, screen_w, screen_h,
            media["width"], media["height"]
        )

        # Discard points outside image bounds
        in_bounds = (
            (img_x >= 0) & (img_x < media["width"])
            & (img_y >= 0) & (img_y < media["height"])
        )
        if not in_bounds.any():
            continue

        pts     = np.column_stack((img_x[in_bounds], img_y[in_bounds]))
        ts_valid = ts_win[in_bounds]

        # AOI hit test
        categories = hit_test(pts, aois)

        # Dwell time: time interval assigned to each sample
        if len(ts_valid) > 1:
            diffs = np.diff(ts_valid)
            median_us = float(np.median(diffs))
            # cap at 200 ms to avoid counting gaps
            diffs = np.minimum(diffs, 200_000)
            sample_dur = np.append(diffs, min(median_us, 200_000))
        else:
            sample_dur = np.array([1_000_000.0 / 60.0])   # assume 60 Hz

        # Accumulate per category
        for cat, dur in zip(categories, sample_dur):
            if cat is None:
                continue
            rows.append({
                "stimulus":   media["name"],
                "category":   cat,
                "dwell_us":   dur,
                "gaze_count": 1,
            })

    if not rows:
        return None

    result = (
        pd.DataFrame(rows)
        .groupby(["stimulus", "category"], as_index=False)
        .agg(dwell_us=("dwell_us", "sum"), gaze_count=("gaze_count", "sum"))
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

    print("=" * 60)
    print("Tobii Studio Attention Analysis")
    print("=" * 60)

    # -- Load lookup tables -------------------------------------------------
    print("\nBuilding stimulus lookup from .exp files...")
    stim_lookup = build_stimulus_lookup(STIMULUS_DIR)
    print(f"  {len(stim_lookup)} stimulus entries")

    print("Loading database (media + AOIs)...")
    media_by_mkey, aoi_by_media = load_database(DB_PATH)
    total_aois = sum(len(v) for v in aoi_by_media.values())
    print(f"  {len(media_by_mkey)} media items, {total_aois} AOI polygons")

    # -- Find recording directories -----------------------------------------
    rec_dirs = sorted(
        d for d in RECORDINGS_DIR.iterdir()
        if d.is_dir() and (d / "Eyetracker.data").exists()
    )
    print(f"  {len(rec_dirs)} recording directories\n")

    # -- Process each recording ---------------------------------------------
    all_results: list[pd.DataFrame] = []
    for idx, rec_dir in enumerate(rec_dirs, 1):
        print(f"[{idx:2d}/{len(rec_dirs)}] {rec_dir.name}")
        result = process_recording(
            rec_dir, stim_lookup, media_by_mkey, aoi_by_media
        )
        if result is not None and not result.empty:
            result.insert(0, "recording", rec_dir.name)
            all_results.append(result)
            n_cat = result["category"].nunique()
            n_stim = result["stimulus"].nunique()
            total_ms = result["dwell_us"].sum() / 1000
            print(f"         {n_stim} stimuli, {n_cat} categories, "
                  f"{total_ms:,.0f} ms total dwell")
        else:
            print("         no usable AOI data")

    if not all_results:
        print("\nNo results collected. Check GUID parsing or data paths.")
        return

    # -- Combine and save outputs -------------------------------------------
    combined = pd.concat(all_results, ignore_index=True)
    combined["dwell_ms"] = (combined["dwell_us"] / 1000).round(1)

    out_raw = OUTPUT_DIR / "attention_by_recording.csv"
    combined.to_csv(out_raw, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_raw.name}  ({len(combined)} rows)")

    # Per stimulus × category
    by_stim = (
        combined.groupby(["stimulus", "category"], as_index=False)
        .agg(
            total_dwell_ms=("dwell_ms", "sum"),
            total_gaze_count=("gaze_count", "sum"),
            n_recordings=("recording", "nunique"),
        )
        .sort_values(["stimulus", "total_dwell_ms"], ascending=[True, False])
    )
    out_stim = OUTPUT_DIR / "attention_by_stimulus.csv"
    by_stim.to_csv(out_stim, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_stim.name}")

    # Per category (aggregate)
    by_cat = (
        combined.groupby("category", as_index=False)
        .agg(
            total_dwell_ms=("dwell_ms", "sum"),
            total_gaze_count=("gaze_count", "sum"),
        )
    )
    total_dwell = by_cat["total_dwell_ms"].sum()
    by_cat["dwell_pct"] = (by_cat["total_dwell_ms"] / total_dwell * 100).round(2)
    by_cat = by_cat.sort_values("total_dwell_ms", ascending=False)

    out_cat = OUTPUT_DIR / "attention_by_category.csv"
    by_cat.to_csv(out_cat, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_cat.name}")

    # -- Console summary ----------------------------------------------------
    print("\n" + "=" * 60)
    print("TOP LANDSCAPE ELEMENTS BY TOTAL DWELL TIME")
    print("=" * 60)
    print(f"  {'Category':<25}  {'Dwell (ms)':>12}  {'% of AOI time':>14}  {'Gaze hits':>10}")
    print("  " + "-" * 65)
    for _, row in by_cat.iterrows():
        print(
            f"  {row['category']:<25}  "
            f"{row['total_dwell_ms']:>12,.0f}  "
            f"{row['dwell_pct']:>13.2f}%  "
            f"{row['total_gaze_count']:>10,.0f}"
        )
    print("=" * 60)
    print(f"  Total AOI dwell: {total_dwell:,.0f} ms across {len(all_results)} recordings")
    print("=" * 60)


if __name__ == "__main__":
    main()
