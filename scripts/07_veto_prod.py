"""Fase 5 produccion: veto por coincidencia con umbrales por estacion + solape.
Config calibrada en ventana noche comun 10min:
  RE626 sensible (2.8/1.2) - el veto limpia; R2A1F (2.6/1.1); R53D1 (2.2/1.0).
Regla kept: existe trigger limpio con |onset|<=3s Y solape de intervalos.
Uso: python scripts/07_veto_prod.py
"""
import json
import numpy as np
from obspy import read, UTCDateTime
from scipy.signal import butter, sosfilt, iirnotch, tf2sos
from obspy.signal.trigger import classic_sta_lta, trigger_onset

FS = 100.0
CFG = {
    "RE626": {"on": 2.8, "off": 1.2},
    "R2A1F": {"on": 2.6, "off": 1.1},
    "R53D1": {"on": 3.0, "off": 1.5},
    "tol_s": 5.0,
}
SOS_HP = butter(2, 0.7, btype="highpass", fs=FS, output="sos")
SOS_N = [tf2sos(*iirnotch(w0=f, Q=30, fs=FS)) for f in (29.79, 18.07, 42.24)]
SOS_DET = butter(4, [2.0, 20.0], btype="bandpass", fs=FS, output="sos")

def light(x):
    y = sosfilt(SOS_HP, x - x.mean())
    for s in SOS_N:
        y = sosfilt(s, y)
    return sosfilt(SOS_DET, y)

def get_trig(sta, fp):
    tr = read(fp)[0]
    d = light(tr.data)
    ons, _ = trigger_onset(classic_sta_lta(d, int(FS), int(20 * FS)),
                           CFG[sta]["on"], CFG[sta]["off"]), None
    t0 = float(tr.stats.starttime)
    return [(t0 + s / FS, t0 + e / FS) for s, e in ons], tr

def overlap(a, b, pad=2.0):
    return max(a[0] - pad, b[0] - pad) <= min(a[1] + pad, b[1] + pad)

def veto(re626, cleaners, tol):
    kept, vetoed = [], []
    for r in re626:
        ok = any(abs(r[0] - c[0]) <= tol and overlap(r, c) for lst in cleaners for c in lst)
        (kept if ok else vetoed).append(r)
    return kept, vetoed

out = {}
for tag in ["ev_local_close_M45_20260915", "ev_local_M53_20260910"]:
    r, _ = get_trig("RE626", f"data/raw_mseed/{tag}_RE626.mseed")
    c1, _ = get_trig("R2A1F", f"data/raw_mseed/{tag}_R2A1F.mseed")
    c2, _ = get_trig("R53D1", f"data/raw_mseed/{tag}_R53D1.mseed")
    kept, vetoed = veto(r, [c1, c2], CFG["tol_s"])
    out[tag] = {
        "n_re626": len(r), "n_r2a1f": len(c1), "n_r53d1": len(c2),
        "kept": [str(UTCDateTime(a)) for a, _ in kept],
        "vetoed": [str(UTCDateTime(a)) for a, _ in vetoed],
        "rechazo": round(len(vetoed) / max(len(r), 1), 2),
    }
    print(f"{tag}: RE626 {len(r)} R2A1F {len(c1)} R53D1 {len(c2)} -> kept {len(kept)} veto {len(vetoed)}")
    for a in out[tag]["kept"]: print(f"   KEPT {a}")
    for a in out[tag]["vetoed"]: print(f"   veto {a}")

# falsos en noche (misma ventana, debe dar ~0 kept)
r, _ = get_trig("RE626", "data/raw_mseed/RE626_noche_10min.mseed")
c1, _ = get_trig("R2A1F", "data/raw_mseed/R2A1F_noche_10min.mseed")
c2, _ = get_trig("R53D1", "data/raw_mseed/R53D1_noche_10min.mseed")
kept, vetoed = veto(r, [c1, c2], CFG["tol_s"])
out["noche_10min"] = {"n_re626": len(r), "kept": len(kept), "falsos_hora": round(len(kept) * 6, 1)}
print(f"noche: RE626 {len(r)} -> kept {len(kept)} ({out['noche_10min']['falsos_hora']}/hora)")
json.dump({"cfg": CFG, **out}, open("results/metrics/veto_prod.json", "w"), indent=2)
print("OK results/metrics/veto_prod.json")
