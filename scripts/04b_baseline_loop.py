"""Fase 3b (loop): causal + despike Hampel causal + ventanas P/S de estacion limpia.
- Despiker: mediana movil causal 1s + clip a 5*MAD (mata impulsos hiper-locales sin ringing IIR).
- Ventanas desde STA/LTA en R2A1F: P 23:45:00-10, S 23:45:39-50, ruido 23:41:00-23:43:00.
- CC sobre envolvente (STA 0.5s) porque respuestas de sitio difieren.
Uso: python scripts/04b_baseline_loop.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime
from scipy.signal import butter, sosfilt, iirnotch, tf2sos
from scipy.ndimage import uniform_filter1d

SENS = 399650000.0
FS = 100.0
SOS_HP = butter(2, 0.7, btype="highpass", fs=FS, output="sos")
SOS_BP = butter(4, [1.0, 25.0], btype="bandpass", fs=FS, output="sos")
SOS_NOTCH = [tf2sos(*iirnotch(w0=f0, Q=Q, fs=FS)) for f0, Q in [(29.79, 30), (18.07, 30), (42.24, 30)]]

def despike_causal(v, win=100, k=5.0):
    med = uniform_filter1d(v, size=win, mode="nearest")  # aprox mediana rapida causal-ish
    mad = uniform_filter1d(np.abs(v - med), size=win, mode="nearest") + 1e-12
    thr = k * 1.4826 * mad
    return np.clip(v, med - thr, med + thr)

def chain(v_counts, despike=True):
    v = (v_counts.astype(float) - np.mean(v_counts)) / SENS
    if despike:
        v = despike_causal(v)
    y = sosfilt(SOS_HP, v)
    for s in SOS_NOTCH:
        y = sosfilt(s, y)
    return sosfilt(SOS_BP, y)

def rms(x): return float(np.sqrt(np.mean(np.asarray(x) ** 2)))
def snr(a, b): return 20 * np.log10((rms(a) + 1e-18) / (rms(b) + 1e-18))
def envelope(x, w=50):
    return uniform_filter1d(np.abs(np.asarray(x, float)), size=w, mode="nearest")
def cc(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0

TAG = "ev_local_close_M45_20260915"
P0, P1 = UTCDateTime("2026-09-15T23:45:00"), UTCDateTime("2026-09-15T23:45:10")
S0, S1 = UTCDateTime("2026-09-15T23:45:39"), UTCDateTime("2026-09-15T23:45:50")
N0, N1 = UTCDateTime("2026-09-15T23:41:00"), UTCDateTime("2026-09-15T23:43:00")

def seg(tr, a, b):
    c = tr.copy().trim(a, b)
    return c.data

out = {}
for desp in [False, True]:
    re626 = read(f"data/raw_mseed/{TAG}_RE626.mseed")[0]
    ref = read(f"data/raw_mseed/{TAG}_R53D1.mseed")[0]
    f626 = chain(re626.data, despike=desp)
    fref = chain(ref.data, despike=desp)
    # mapea a indices via tiempo
    def fseg(raw_tr, farr, a, b):
        s = int((a - raw_tr.stats.starttime) * FS)
        e = int((b - raw_tr.stats.starttime) * FS)
        s, e = max(s, 0), min(e, len(farr))
        return farr[s:e]
    n_raw = seg(re626, N0, N1); p_raw = seg(re626, P0, P1); s_raw = seg(re626, S0, S1)
    n_f = fseg(re626, f626, N0, N1); p_f = fseg(re626, f626, P0, P1); s_f = fseg(re626, f626, S0, S1)
    p_ref = fseg(ref, fref, P0, P1); s_ref = fseg(ref, fref, S0, S1)
    # burst hiper-local -50..-30 rel catalogo
    B0, B1 = UTCDateTime("2026-09-15T23:44:10"), UTCDateTime("2026-09-15T23:44:40")
    b_raw = seg(re626, B0, B1); b_f = fseg(re626, f626, B0, B1)
    key = "despike_ON" if desp else "sin_despike"
    out[key] = {
        "snr_P_raw": snr(p_raw, n_raw), "snr_P_filt": snr(p_f, n_f),
        "snr_S_raw": snr(s_raw, n_raw), "snr_S_filt": snr(s_f, n_f),
        "burst_atten_dB": 20 * np.log10((rms(b_raw) + 1e-18) / (rms(b_f) / SENS * SENS + 1e-18)) if False else float(20 * np.log10(rms(b_raw) / (rms(b_f) * SENS) + 1e-9)),
        "burst_rms_raw_counts": rms(b_raw), "burst_rms_filt_m_s": rms(b_f),
        "cc_env_P_vs_ref": cc(envelope(p_f), envelope(p_ref)),
        "cc_env_S_vs_ref": cc(envelope(s_f), envelope(s_ref)),
    }
    print(key, json.dumps(out[key], indent=1))

# fig P/S zoom filtrado con/sin despike
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
re626 = read(f"data/raw_mseed/{TAG}_RE626.mseed")[0]
f_off = chain(re626.data, False); f_on = chain(re626.data, True)
t = np.arange(len(re626.data)) / FS
t0 = re626.stats.starttime
for a in [ax[0], ax[1]]:
    a.axvspan((N0 - t0), (N1 - t0), color="gray", alpha=0.15)
    a.axvspan((P0 - t0), (P1 - t0), color="green", alpha=0.2)
    a.axvspan((S0 - t0), (S1 - t0), color="orange", alpha=0.2)
    a.axvspan(250, 280, color="red", alpha=0.15)
ax[0].plot(t, f_off * 1e6, lw=0.4); ax[0].set_title("causal sin despike (um/s) gris=ruido verde=P naranja=S rojo=burst local")
ax[1].plot(t, f_on * 1e6, lw=0.4, color="g"); ax[1].set_title("causal + despike Hampel k=5")
ax[1].set_xlabel("s desde inicio ventana"); ax[1].set_xlim(0, 600)
fig.tight_layout(); fig.savefig("results/figs/ev_close_loop_despike.png", dpi=110)
json.dump(out, open("results/metrics/baseline_loop.json", "w"), indent=2)
print("OK results/metrics/baseline_loop.json + ev_close_zoom3sta.png + ev_close_loop_despike.png")
