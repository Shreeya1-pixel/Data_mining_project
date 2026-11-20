"""Sample GDELT 2.0 GKG files for additional experiment months (pilot-compatible schema)."""

from __future__ import annotations

import csv
import io
import json
import random
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

csv.field_size_limit(sys.maxsize)

GEO_THEMES = {
    "CONFLICT": ("ARMEDCONFLICT", "MILITARY", "KILL", "WMD", "MIL_", "CEASEFIRE", "REBELLION"),
    "TERROR": ("TERROR", "SUICIDE_ATTACK"),
    "SANCTION": ("SANCTIONS", "EMBARGO"),
    "TRADE": ("ECON_TRADE", "TAX_TARIFF", "ECON_FREETRADE", "WTO"),
    "ENERGY": ("ENV_OIL", "ECON_OILPRICE", "ENV_NATURALGAS", "OPEC", "FUELPRICES"),
    "SHIPPING": ("MARITIME", "BLOCKADE", "SUPPLY_CHAIN", "PIRACY"),
    "DIPLO": ("DIPLOMACY", "BORDER", "TERRITORIAL", "NEGOTIATIONS"),
    "POLITICS": ("ELECTION", "COUP", "PROTEST", "POLITICAL_TURMOIL"),
    "MONETARY": ("ECON_CENTRALBANK", "ECON_INTEREST_RATE", "ECON_INFLATION", "ECON_CURRENCY"),
    "TECH": ("CYBER_ATTACK", "CYBERATTACK", "TECH_", "EXPORT_CONTROL"),
}
TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)


def slots(year: int, month: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    d = date(year, month, 1)
    out = []
    while d.month == month:
        for block in range(4):
            q = rng.randrange(24)
            h, m = block * 6 + q // 4, (q % 4) * 15
            out.append(f"{d:%Y%m%d}{h:02d}{m:02d}00")
        d += timedelta(days=1)
    return out


def _india(themes: str, locations: str) -> bool:
    blob = (themes or "") + " " + (locations or "")
    return "INDIA" in blob.upper() or "IN#" in blob.upper() or "New Delhi" in blob


def process(stamp: str) -> list[dict]:
    url = f"http://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip"
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        raw = z.read(z.namelist()[0])
    except Exception:
        return []
    keep = []
    for row in csv.reader(io.StringIO(raw.decode("utf-8", "replace")), delimiter="\t"):
        if len(row) < 27:
            continue
        themes, locations, tone, source, url_s, extras = row[7], row[9], row[15], row[3], row[4], row[26]
        if not any(any(p in themes for p in pats) for pats in GEO_THEMES.values()):
            continue
        if not (_india(themes, locations) or "ECON_" in themes):
            continue
        m = TITLE_RE.search(extras or "")
        title = (m.group(1).strip() if m else "")[:300]
        try:
            tone_f = float(str(tone).split(",")[0])
        except Exception:
            tone_f = 0.0
        keep.append(
            {
                "date": stamp,
                "source": (source or "").lower().replace("www.", ""),
                "url": url_s,
                "title": title,
                "themes": themes,
                "tone": tone_f,
                "india": _india(themes, locations),
            }
        )
    return keep


def harvest_month(year: int, month: int, out_dir: Path, workers: int = 8) -> Path:
    tag = f"{year}{month:02d}"
    out = out_dir / f"gkg_pilot_rows_{tag}.parquet"
    if out.exists():
        return out
    stamps = slots(year, month)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process, s): s for s in stamps}
        for i, fut in enumerate(as_completed(futs), 1):
            rows.extend(fut.result())
            if i % 20 == 0:
                print(f"{tag}: {i}/{len(stamps)} files, {len(rows)} rows", flush=True)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    (out_dir / f"gkg_pilot_summary_{tag}.json").write_text(
        json.dumps({"tag": tag, "n_rows": len(df), "n_files": len(stamps)}, indent=2)
    )
    print(f"wrote {out} ({len(df)} rows)")
    return out


if __name__ == "__main__":
    months = [tuple(map(int, m.split("-"))) for m in sys.argv[1:]] or [
        (2022, 2),
        (2023, 10),
        (2024, 4),
        (2025, 5),
    ]
    dest = Path(__file__).resolve().parents[1] / "data" / "pilot"
    for y, m in months:
        harvest_month(y, m, dest)
