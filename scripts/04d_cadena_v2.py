"""Fase 3 fix: cadena v2 causal = HP0.7 + notch x3 + BP 1-40 + gating Wiener causal.
Gating: PSD ruido de ventana previa 120s (causal, solo pasado), ganancia por frame 10s.
Uso: python scripts/04d_cadena_v2.py
"""
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime
from scipy.signal import butter, sosfilt, iirnotch, tf2sos, welch

SENS = 399650000.0
FS = 100.0
SOS_HP = butter(2, 0.7, btype="highpass", fs=FS, output="sos")
SOS_N = [tf2sos(*iirnotch(w0=f, Q=30, fs=FS)) for f in (29.79, 18.07, 42.24)]
SOS_BP40 = butter(4, [1.0, 40.0], btype="bandpass", fs=FS, output="sos")

def pre(v):
    v = (v.astype(float) - v.mean()) / SENS
    y = sosfilt(SOS_HP, v)
    for s in SOS_N:
        y = sosfilt(s, y)
    return sosfilt(SOS_BP40, y)

def causal_gating(y, fs=FS, frame=1000, alpha=2.0, floor=0.1):
    """y: senal prefiltrada. Ruido = primeros 120s (causal: solo pasado de P/S)."""
    nref = y[:120 * int(fs)]
    f, Pnn = welch(nref, fs=fs, nperseg=512, noverlap=256)
    out = np.zeros_like(y)
    for i in range(0, len(y), frame):
        seg = y[i:i + frame]
        X = np.fft.rfft(seg)
        fr = np.fft.rfftfreq(len(seg), 1 / fs)
        Pn = np.interp(fr, f, Pnn)
        Px = (np.abs(X) ** 2) / max(len(seg), 1)
        G = np.clip((Px - alpha * Pn) / (Px + 1e-18), floor, 1.0)
        out[i:i + frame] = np.fft.irfft(X * G, len(seg))
    return out

def rms(x): return float(np.sqrt(np.mean(np.asarray(x) ** 2)))
def snr(a, b): return 20 * np.log10((rms(a) + 1e-18) / (rms(b) + 1e-18))

TAG = "ev_local_close_M45_20260915"
P0, P1 = UTCDateTime("2026-09-15T23:45:00"), UTCDateTime("2026-09-15T23:45:10")
S0, S1 = UTCDateTime("2026-09-15T23:45:39"), UTCDateTime("2026-09-15T23:45:50")
N0, N1 = UTCDateTime("2026-09-15T23:41:00"), UTCDateTime("2026-09-15T23:43:00")
tr = read(f"data/raw_mseed/{TAG}_RE626.mseed")[0]
t = time.perf_counter()
y = pre(tr.data)
yg = causal_gating(y)
lat = (time.perf_counter() - t) * 1000
def seg(arr, a, b):
    s = max(int((a - tr.stats.starttime) * FS), 0)
    return arr[s:min(int((b - tr.stats.starttime) * FS), len(arr))]
n0, p0, s0 = seg(tr.data / SENS, N0, N1), seg(tr.data / SENS, P0, P1), seg(tr.data / SENS, S0, S1)
n1, p1, s1 = seg(y, N0, N1), seg(y, P0, P1), seg(y, S0, S1)
n2, p2, s2 = seg(yg, N0, N1), seg(yg, P0, P1), seg(yg, S0, S1)
m = {
    "snrP_raw": snr(p0, n0), "snrS_raw": snr(s0, n0),
    "snrP_prefilt_1_40": snr(p1, n1), "snrS_prefilt_1_40": snr(s1, n1),
    "snrP_v2_gating": snr(p2, n2), "snrS_v2_gating": snr(s2, n2),
    "latency_ms_600s": lat, "latency_per_60s_ms": lat / 10,
}
print(json.dumps(m, indent=1))
json.dump(m, open("results/metrics/cadena_v2.json", "w"), indent=2)
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
tt = np.arange(len(y)) / FS
ax[0].plot(tt, y * 1e6, lw=0.4); ax[0].set_title("v2 pre (HP+notch+BP1-40) um/s")
for a, b, c in [(N0, N1, "gray"), (P0, P1, "green"), (S0, S1, "orange")]:
    for x in ax:
        x.axvspan((a - tr.stats.starttime), (b - tr.stats.starttime), color=c, alpha=0.15)
ax[1].plot(tt, yg * 1e6, lw=0.4, color="g"); ax[1].set_title("v2 + gating Wiener causal")
ax[1].set_xlabel("s"); ax[1].set_xlim(0, 600)
fig.tight_layout(); fig.savefig("results/figs/cadena_v2.png", dpi=110)
print("OK results/metrics/cadena_v2.json + results/figs/cadena_v2.png")
