"""Fase 2: dataset benchmark - 4 eventos (2 locales, 1 regional, 1 tele) +-300s.
RE626 + 3 candidatas limpias. Guarda solo las que tengan datos.
Uso: python scripts/03_download_events.py
"""
import os
from obspy import UTCDateTime
from rs_fdsn import dataselect

OUT = "data/raw_mseed"
os.makedirs(OUT, exist_ok=True)

EVENTS = {
    "ev_local_close_M45_20260915": "2026-09-15T23:45:06",
    "ev_local_M53_20260910": "2026-09-10T06:42:40",
    "ev_regional_M45_20260830": "2026-08-30T04:37:56",
    "ev_tele_M67_20260820": "2026-08-20T18:00:18",
}
STATIONS = ["RE626", "R2A1F", "R0949", "R53D1"]
HALF = 300  # +-300s

def dl(sta, t0, t1, tag):
    fp = f"{OUT}/{tag}_{sta}.mseed"
    if os.path.exists(fp) and os.path.getsize(fp) > 1000:
        print(f"  {sta}: existe, skip")
        return True
    try:
        data, _ = dataselect("AM", sta, "00", "EHZ", t0, t1)
        if len(data) < 1000:
            print(f"  {sta}: vacio")
            return False
        open(fp, "wb").write(data)
        print(f"  {sta}: {len(data)} bytes OK")
        return True
    except Exception as e:
        print(f"  {sta}: FAIL {str(e)[:120]}")
        return False

def main():
    ok = {}
    for tag, tcenter in EVENTS.items():
        print(f"== {tag} @ {tcenter} ==")
        t = UTCDateTime(tcenter)
        t0 = (t - HALF).strftime("%Y-%m-%dT%H:%M:%S")
        t1 = (t + HALF).strftime("%Y-%m-%dT%H:%M:%S")
        ok[tag] = [s for s in STATIONS if dl(s, t0, t1, tag)]
    print("\nResumen:")
    for k, v in ok.items():
        print(f"  {k}: {v}")
    # guarda lista para fase 3
    import json
    json.dump(ok, open("data/raw_mseed/_events_ok.json", "w"), indent=2)

if __name__ == "__main__":
    main()
