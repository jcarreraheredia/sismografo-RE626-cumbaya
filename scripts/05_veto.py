"""Fase 5: veto por coincidencia RE626 vs R2A1F/R53D1.
Cadena ligera (HP+notch, sin BP agresivo) + STA/LTA 1s/20s sobre BP 2-20 solo para deteccion.
Regla: trigger RE626 valido solo si >=1 limpia triggerea en +-5s.
Evalua: burst -40s (debe vetarse) y P/S catalogo (deben pasar) + ventana ruido (falsos).
Uso: python scripts/05_veto.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime
from scipy.signal import butter, sosfilt, iirnotch, tf2sos
from obspy.signal.trigger import classic_sta_lta, trigger_onset

SENS = {"RE626": 399650000.0, "R2A1F": 399650000.0, "R53D1": 399650000.0}
FS = 100.0
SOS_HP = butter(2, 0.7, btype="highpass", fs=FS, output="sos")
SOS_N = [tf2sos(*iirnotch(w0=f, Q=30, fs=FS)) for f in (29.79, 18.07, 42.24)]
SOS_DET = butter(4, [2.0, 20.0], btype="bandpass", fs=FS, output="sos")

def light(counts):
    v = (counts.astype(float) - counts.mean())
    y = sosfilt(SOS_HP, v)
    for s in SOS_N:
        y = sosfilt(s, y)
    return sosfilt(SOS_DET, y)

def triggers(det, thr_on=2.8, thr_off=1.2):
    cft = classic_sta_lta(det, int(1 * FS), int(20 * FS))
    return trigger_onset(cft, thr_on, thr_off), cft

TAG = "ev_local_close_M45_20260915"
stas = {}
for sta in ["RE626", "R2A1F", "R53D1"]:
    tr = read(f"data/raw_mseed/{TAG}_{sta}.mseed")[0]
    d = light(tr.data)
    ons, cft = triggers(d)
    t0 = tr.stats.starttime
    times = [(float(t0 + s / FS), float(t0 + e / FS)) for s, e in ons]
    stas[sta] = {"t0": t0, "tr": tr, "det": d, "cft": cft, "trig": times}
    print(f"{sta}: {len(ons)} triggers")
    for a, b in times[:8]:
        print(f"   {UTCDateTime(a)} -> {UTCDateTime(b)}")

# coincidencia
def coinc(t, others, tol=5.0):
    return any(abs(t - o) <= tol for lst in others for (o, _) in lst)

re626 = stas["RE626"]["trig"]
others = [stas["R2A1F"]["trig"], stas["R53D1"]["trig"]]
kept, vetoed = [], []
for a, b in re626:
    (kept if coinc(a, others) else vetoed).append((a, b))
print(f"\nRE626: {len(re626)} triggers -> kept {len(kept)} / vetoed {len(vetoed)}")
for a, b in kept:
    print(f"  KEPT {UTCDateTime(a)}")
for a, b in vetoed:
    print(f"  VETO {UTCDateTime(a)}")

# ventanas de verdad
P = UTCDateTime("2026-09-15T23:45:00")
BURST = UTCDateTime("2026-09-15T23:44:26")
def near(lst, t, tol=15): return [x for x in lst if abs(x[0] - float(t)) <= tol]
print("\nP real:", [str(UTCDateTime(a)) for a, _ in near(kept, P)], "(debe haber >=1 KEPT)")
print("Burst:", [str(UTCDateTime(a)) for a, _ in near(vetoed, BURST)], "(debe estar VETO)")

json.dump({k: {"ntrig": len(v["trig"]), "trig_utc": [str(UTCDateTime(a)) for a, _ in v["trig"]]}
           for k, v in stas.items()} | {"kept": [str(UTCDateTime(a)) for a, _ in kept],
                                        "vetoed": [str(UTCDateTime(a)) for a, _ in vetoed]},
          open("results/metrics/veto.json", "w"), indent=2)

# fig CFT
fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
for i, sta in enumerate(["RE626", "R2A1F", "R53D1"]):
    v = stas[sta]
    t = np.arange(len(v["cft"])) / FS
    ax[i].plot(t, v["cft"], lw=0.5)
    ax[i].axhline(3.5, color="r", ls="--", lw=0.8)
    ax[i].set_title(f"{sta} STA/LTA (thr 3.5)")
    for a, b in v["trig"]:
        s = a - float(v["t0"])
        ax[i].axvspan(s, b - float(v["t0"]), color="red" if sta == "RE626" and (a, b) in vetoed else "green", alpha=0.25)
ax[2].set_xlabel("s desde inicio")
fig.tight_layout(); fig.savefig("results/figs/veto_cft.png", dpi=110)
print("OK results/metrics/veto.json + results/figs/veto_cft.png")
