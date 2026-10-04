"""Pipeline unificado produccion (Fase 8): pre -> STA/LTA -> veto -> (PhaseNet en kept).
Modos: historico (mseed/FDSN, actual) y plantilla live (SeedLink/rsudp, requiere Shake).
Uso historico: python scripts/pipeline.py --mseed data/raw_mseed/ev_local_close_M45_20260915_RE626.mseed
Uso live (plantilla): python scripts/pipeline.py --live --host data.raspberryshake.org
"""
import argparse, time, yaml
import numpy as np
from scipy.signal import butter, sosfilt, iirnotch, tf2sos

CFG = yaml.safe_load(open("config.yaml"))
FS = float(CFG["estacion_objetivo"]["fs"])
SENS = float(CFG["estacion_objetivo"]["sens_counts_per_ms"])
SOS_HP = butter(CFG["prefiltro"]["orden_hp"], CFG["prefiltro"]["highpass"],
                btype="highpass", fs=FS, output="sos")
SOS_N = [tf2sos(*iirnotch(w0=f, Q=CFG["prefiltro"]["notch_Q"], fs=FS))
         for f in CFG["prefiltro"]["notch"]]
lo, hi = CFG["deteccion"]["banda"]
SOS_DET = butter(CFG["deteccion"]["orden_bp"], [lo, hi], btype="bandpass", fs=FS, output="sos")

def prefiltrar(counts):
    v = (np.asarray(counts, float) - np.mean(counts)) / SENS
    y = sosfilt(SOS_HP, v)
    for s in SOS_N:
        y = sosfilt(s, y)
    return y  # sin BP agresivo (Fase 3/3c: dana P)

def deteccion(counts):
    from obspy.signal.trigger import classic_sta_lta, trigger_onset
    y = sosfilt(SOS_DET, prefiltrar(counts) * SENS)  # deteccion en counts
    cft = classic_sta_lta(y, int(CFG["deteccion"]["sta_s"] * FS),
                          int(CFG["deteccion"]["lta_s"] * FS))
    return trigger_onset(cft, 2.8, 1.2), cft  # umbrales RE626 por defecto; ver CFG por estacion

def veto(re626, cleaners, tol=None, pad=None):
    tol = tol or CFG["veto"]["tol_onset_s"]
    pad = pad or CFG["veto"]["pad_solape_s"]
    kept, vetoed = [], []
    for r in re626:
        ok = any(abs(r[0] - c[0]) <= tol and
                 max(r[0] - pad, c[0] - pad) <= min(r[1] + pad, c[1] + pad)
                 for lst in cleaners for c in lst)
        (kept if ok else vetoed).append(r)
    return kept, vetoed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mseed", default=None)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--host", default="data.raspberryshake.org")
    a = ap.parse_args()
    if a.live:
        print("MODO LIVE no disponible sin acceso al Shake (ver Fase 8 pendiente).")
        print(f"Plantilla: SeedLink {a.host} o rsudp UDP local; misma cadena pre->STA/LTA->veto.")
        print("Requiere: IP/puerto del Shake + reloj NTP + cluster online R2A1F/R53D1.")
        return 2
    if not a.mseed:
        print("Uso: pipeline.py --mseed <archivo>  |  pipeline.py --live")
        return 2
    from obspy import read
    t = time.perf_counter()
    tr = read(a.mseed)[0]
    ons, _ = deteccion(tr.data)
    dt = (time.perf_counter() - t) * 1000
    t0 = float(tr.stats.starttime)
    print(f"{tr.id} n={len(tr.data)} triggers={len(ons)} lat={dt:.1f}ms")
    for s, e in ons:
        print(f"  {tr.stats.starttime + s / FS} -> {tr.stats.starttime + e / FS}")
    print("NOTA: veto y pick requieren Streams del cluster (ver 07_veto_prod.py y 08_fase6_picker.py).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
