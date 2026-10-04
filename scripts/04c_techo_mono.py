"""Fase 3c: techo mono-estacion. Grid bandas + Wiener ideal offline.
Si ni el Wiener ideal supera +3dB, Fase 3 queda cerrada y se avanza a multi-estacion.
Uso: python scripts/04c_techo_mono.py
"""
import json
import numpy as np
from obspy import read, UTCDateTime
from scipy.signal import butter, sosfilt, iirnotch, tf2sos, welch, get_window

SENS = 399650000.0
FS = 100.0
NOTCH = [(29.79, 30), (18.07, 30), (42.24, 30)]
SOS_N = [tf2sos(*iirnotch(w0=f, Q=q, fs=FS)) for f, q in NOTCH]
SOS_HP = butter(2, 0.7, btype="highpass", fs=FS, output="sos")

def base(v):
    v = (v.astype(float) - v.mean()) / SENS
    y = sosfilt(SOS_HP, v)
    for s in SOS_N:
        y = sosfilt(s, y)
    return y

def bp(y, lo, hi):
    return sosfilt(butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos"), y)

def rms(x): return float(np.sqrt(np.mean(np.asarray(x) ** 2)))
def snr(a, b): return 20 * np.log10((rms(a) + 1e-18) / (rms(b) + 1e-18))

def wiener_gain(noise, sigwin, nperseg=512):
    nov_n = nperseg // 2
    f, Pnn = welch(noise, fs=FS, nperseg=nperseg, noverlap=nov_n)
    ns = min(nperseg, len(sigwin))
    _, Pss = welch(sigwin, fs=FS, nperseg=ns, noverlap=ns // 2)
    G = np.clip((Pss - Pnn) / (Pss + 1e-18), 0.05, 1.0)
    return f, G

def apply_gain(x, f, G):
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), 1 / FS)
    Gi = np.interp(fr, f, G)
    return np.fft.irfft(X * Gi, len(x))

TAG = "ev_local_close_M45_20260915"
P0, P1 = UTCDateTime("2026-09-15T23:45:00"), UTCDateTime("2026-09-15T23:45:10")
S0, S1 = UTCDateTime("2026-09-15T23:45:39"), UTCDateTime("2026-09-15T23:45:50")
N0, N1 = UTCDateTime("2026-09-15T23:41:00"), UTCDateTime("2026-09-15T23:43:00")
tr = read(f"data/raw_mseed/{TAG}_RE626.mseed")[0]
v = base(tr.data)
def seg(a, b):
    s = max(int((a - tr.stats.starttime) * FS), 0)
    e = min(int((b - tr.stats.starttime) * FS), len(v))
    return v[s:e]
n, p, s = seg(N0, N1), seg(P0, P1), seg(S0, S1)

print(f"base sin BP: SNR_P={snr(p,n):.2f} SNR_S={snr(s,n):.2f}")
res = {}
for lo, hi in [(1, 25), (2, 15), (2, 20), (3, 15), (1, 10)]:
    pf, sf, nf = bp(p, lo, hi), bp(s, lo, hi), bp(n, lo, hi)
    res[f"BP_{lo}_{hi}"] = {"snrP": snr(pf, nf), "snrS": snr(sf, nf)}
    print(f"BP {lo}-{hi}: SNR_P={snr(pf,nf):.2f} SNR_S={snr(sf,nf):.2f}")

# Wiener ideal: mascara desde el propio P y S (techo inalcanzable en vivo = cota superior)
for name, sig in [("P", p), ("S", s)]:
    f, G = wiener_gain(n, sig)
    pw, sw, nw = apply_gain(p, f, G), apply_gain(s, f, G), apply_gain(n, f, G)
    res[f"wiener_ideal_{name}"] = {"snrP": snr(pw, nw), "snrS": snr(sw, nw)}
    print(f"Wiener ideal ({name}): SNR_P={snr(pw,nw):.2f} SNR_S={snr(sw,nw):.2f}")

json.dump(res, open("results/metrics/techo_mono.json", "w"), indent=2)
print("OK results/metrics/techo_mono.json")
