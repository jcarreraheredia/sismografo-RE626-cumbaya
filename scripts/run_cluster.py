"""Runner cluster escalable: lee config.yaml, detecta por estacion, aplica veto.
Misma funcion en laptop y en edge futuro (solo cambia fuente de datos).
Uso: python scripts/run_cluster.py --tag hour1 [--live-proximo]
"""
import argparse, json, time
import numpy as np
import yaml
from obspy import read
from scipy.signal import butter, sosfilt, iirnotch, tf2sos
from obspy.signal.trigger import classic_sta_lta, trigger_onset

CFG = yaml.safe_load(open("config.yaml"))
FS = float(CFG["estacion_objetivo"]["fs"])
SENS = float(CFG["estacion_objetivo"]["sens_counts_per_ms"])
SOS_HP = butter(CFG["prefiltro"]["orden_hp"], CFG["prefiltro"]["highpass"],
                btype="highpass", fs=FS, output="sos")
SOS_N = [tf2sos(*iirnotch(w0=f, Q=CFG["prefiltro"]["notch_Q"], fs=FS))
         for f in CFG["prefiltro"]["notch"]]
lo, hi = CFG["deteccion"]["banda"]
SOS_DET = butter(CFG["deteccion"]["orden_bp"], [lo, hi], btype="bandpass", fs=FS, output="sos")
THR = CFG["deteccion"]["umbrales"]
TOL = CFG["veto"]["tol_onset_s"]
PAD = CFG["veto"]["pad_solape_s"]

def light(x):
    y = sosfilt(SOS_HP, x - x.mean())
    for s in SOS_N:
        y = sosfilt(s, y)
    return sosfilt(SOS_DET, y)

def detect(sta, x):
    cft = classic_sta_lta(light(x), int(CFG["deteccion"]["sta_s"] * FS),
                          int(CFG["deteccion"]["lta_s"] * FS))
    return trigger_onset(cft, THR[sta]["on"], THR[sta]["off"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hour1")
    a = ap.parse_args()
    t = time.perf_counter()
    trig, t0 = {}, {}
    for sta in ["RE626", "R2A1F", "R53D1"]:
        tr = read(f"data/raw_mseed/{a.tag}_{sta}.mseed")[0]
        t0[sta] = float(tr.stats.starttime)
        trig[sta] = [(t0[sta] + s / FS, t0[sta] + e / FS) for s, e in detect(sta, tr.data)]
        print(f"{sta}: {len(trig[sta])} triggers")
    r = trig["RE626"]
    kept = [x for x in r if any(abs(x[0] - c[0]) <= TOL and
            max(x[0] - PAD, c[0] - PAD) <= min(x[1] + PAD, c[1] + PAD)
            for lst in (trig["R2A1F"], trig["R53D1"]) for c in lst)]
    vetoed = [x for x in r if x not in kept]
    dur_h = 1.0
    from obspy import UTCDateTime
    print(f"RE626 {len(r)} -> kept {len(kept)} veto {len(vetoed)} | falsos/hora(kept)={len(kept)/dur_h:.1f}")
    for x in kept: print(f"  KEPT {UTCDateTime(x[0])}")
    for x in vetoed: print(f"  veto {UTCDateTime(x[0])}")
    print(f"lat total {(time.perf_counter()-t)*1000:.0f}ms")
    json.dump({"tag": a.tag, "cfg_thr": THR, "tol": TOL, "pad": PAD,
               "n": {k: len(v) for k, v in trig.items()},
               "kept": [str(UTCDateTime(x[0])) for x in kept],
               "vetoed": [str(UTCDateTime(x[0])) for x in vetoed],
               "falsos_hora": len(kept) / dur_h},
              open(f"results/metrics/cluster_{a.tag}.json", "w"), indent=2)

if __name__ == "__main__":
    main()
