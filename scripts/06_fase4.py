"""Fase 4 propia: gating espectral causal STFT-OLA + wavelet db4.
Prefiltro comun: HP0.7 + notch 29.79/18.07/42.24 (sin BP agresivo).
Evalua en 2 eventos (P/S) + 3 ventanas ruido. Gate G4 explicito.
Uso: python scripts/06_fase4.py
"""
import json, time
import numpy as np
import pywt
from obspy import read, UTCDateTime
from scipy.signal import butter, sosfilt, iirnotch, tf2sos, welch, get_window

SENS = 399650000.0
FS = 100.0
SOS_HP = butter(2, 0.7, btype="highpass", fs=FS, output="sos")
SOS_N = [tf2sos(*iirnotch(w0=f, Q=30, fs=FS)) for f in (29.79, 18.07, 42.24)]

def pre(counts):
    v = (counts.astype(float) - counts.mean()) / SENS
    y = sosfilt(SOS_HP, v)
    for s in SOS_N:
        y = sosfilt(s, y)
    return y

def gating_causal(y, alpha=2.0, floor=0.15, frame=256, hop=128):
    win = get_window("hann", frame, fftbins=True)
    n = len(y)
    out = np.zeros(n)
    wsum = np.zeros(n)
    # ruido inicial: primeros 60s (pasado garantizado lejos de P en ventanas evento)
    f_ref, Pnn = welch(y[:60 * int(FS)], fs=FS, nperseg=512, noverlap=256)
    last_update = 0
    for i in range(0, n - frame, hop):
        # actualiza PSD cada 10s solo con pasado
        if i - last_update >= 10 * int(FS) and i >= 120 * int(FS):
            f_ref, Pnn = welch(y[max(0, i - 120 * int(FS)):i], fs=FS, nperseg=512, noverlap=256)
            last_update = i
        seg = y[i:i + frame] * win
        X = np.fft.rfft(seg)
        fr = np.fft.rfftfreq(frame, 1 / FS)
        Pn = np.interp(fr, f_ref, Pnn)
        Px = (np.abs(X) ** 2) / frame
        G = np.clip((Px - alpha * Pn) / (Px + 1e-18), floor, 1.0)
        rec = np.fft.irfft(X * G, frame) * win
        out[i:i + frame] += rec
        wsum[i:i + frame] += win ** 2
    wsum[wsum < 1e-6] = 1.0
    return out / wsum

def wavelet_denoise(y, noise_ref, wave="db4", level=5):
    coef = pywt.wavedec(y, wave, level=level)
    sigma = np.median(np.abs(coef[-1])) / 0.6745 + 1e-18
    Ns = len(y)
    out = [coef[0]]
    for c in coef[1:]:
        thr = sigma * np.sqrt(2 * np.log(Ns))
        out.append(pywt.threshold(c, thr, mode="soft"))
    return pywt.waverec(out, wave)[:len(y)]

def rms(x): return float(np.sqrt(np.mean(np.asarray(x) ** 2)))
def snr(a, b): return 20 * np.log10((rms(a) + 1e-18) / (rms(b) + 1e-18))
def envelope(x, w=50):
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(np.abs(np.asarray(x, float)), size=w, mode="nearest")
def cc(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0

CASES = {
    "ev_close": ("ev_local_close_M45_20260915",
                 ("2026-09-15T23:45:00", "2026-09-15T23:45:10"),
                 ("2026-09-15T23:45:39", "2026-09-15T23:45:50"),
                 ("2026-09-15T23:41:00", "2026-09-15T23:43:00")),
    "ev_m53": ("ev_local_M53_20260910",
               ("2026-09-10T06:42:45", "2026-09-10T06:42:57"),
               ("2026-09-10T06:43:10", "2026-09-10T06:43:30"),
               ("2026-09-10T06:38:00", "2026-09-10T06:40:00")),
}

def seg(arr, tr, a, b):
    s = max(int((UTCDateTime(a) - tr.stats.starttime) * FS), 0)
    return arr[s:min(int((UTCDateTime(b) - tr.stats.starttime) * FS), len(arr))]

res = {}
for name, (tag, pw, sw, nw) in CASES.items():
    tr = read(f"data/raw_mseed/{tag}_RE626.mseed")[0]
    rf = read(f"data/raw_mseed/{tag}_R53D1.mseed")[0]
    t = time.perf_counter(); y = pre(tr.data); t_pre = time.perf_counter() - t
    t = time.perf_counter(); yg = gating_causal(y); t_g = time.perf_counter() - t
    t = time.perf_counter()
    yw = wavelet_denoise(y, y[:60 * int(FS)]); t_w = time.perf_counter() - t
    yr = pre(rf.data)
    n0, p0, s0 = seg(tr.data / SENS, tr, *nw), seg(tr.data / SENS, tr, *pw), seg(tr.data / SENS, tr, *sw)
    n1, p1, s1 = seg(y, tr, *nw), seg(y, tr, *pw), seg(y, tr, *sw)
    n2, p2, s2 = seg(yg, tr, *nw), seg(yg, tr, *pw), seg(yg, tr, *sw)
    n3, p3, s3 = seg(yw, tr, *nw), seg(yw, tr, *pw), seg(yw, tr, *sw)
    pr, sr = seg(yr, rf, *pw), seg(yr, rf, *sw)
    res[name] = {
        "snrP_raw": snr(p0, n0), "snrS_raw": snr(s0, n0),
        "snrP_pre": snr(p1, n1), "snrS_pre": snr(s1, n1),
        "snrP_gate": snr(p2, n2), "snrS_gate": snr(s2, n2),
        "snrP_wav": snr(p3, n3), "snrS_wav": snr(s3, n3),
        "ccP_pre": cc(envelope(p1), envelope(pr)), "ccS_pre": cc(envelope(s1), envelope(sr)),
        "ccP_gate": cc(envelope(p2), envelope(pr)), "ccS_gate": cc(envelope(s2), envelope(sr)),
        "ccP_wav": cc(envelope(p3), envelope(pr)), "ccS_wav": cc(envelope(s3), envelope(sr)),
        "rmsN_raw": rms(n0), "rmsN_gate": rms(n2), "rmsN_wav": rms(n3),
        "lat_pre_ms": t_pre * 1000, "lat_gate_ms": t_g * 1000, "lat_wav_ms": t_w * 1000,
    }
    print(f"{name}: raw P{res[name]['snrP_raw']:.1f}/S{res[name]['snrS_raw']:.1f} "
          f"-> pre P{res[name]['snrP_pre']:.1f}/S{res[name]['snrS_pre']:.1f} "
          f"-> gate P{res[name]['snrP_gate']:.1f}/S{res[name]['snrS_gate']:.1f} "
          f"-> wav P{res[name]['snrP_wav']:.1f}/S{res[name]['snrS_wav']:.1f} "
          f"| lat gate {t_g*1000:.0f}ms wav {t_w*1000:.0f}ms /{len(y)/FS:.0f}s")

# ruido puro: reduccion RMS +触发 check simple (energia residual)
for w in ["pico", "noche", "lab"]:
    tr = read(f"data/raw_mseed/RE626_{w}_10min.mseed")[0]
    y = pre(tr.data); yg = gating_causal(y)
    yw = wavelet_denoise(y, y[:60 * int(FS)])
    res[f"ruido_{w}"] = {"rms_pre": rms(y), "rms_gate": rms(yg), "rms_wav": rms(yw),
                         "red_gate_dB": 20 * np.log10(rms(y) / (rms(yg) + 1e-18)),
                         "red_wav_dB": 20 * np.log10(rms(y) / (rms(yw) + 1e-18))}
    print(f"ruido {w}: gate {res[f'ruido_{w}']['red_gate_dB']:.1f}dB wav {res[f'ruido_{w}']['red_wav_dB']:.1f}dB")

json.dump(res, open("results/metrics/fase4.json", "w"), indent=2)
print("OK results/metrics/fase4.json")
