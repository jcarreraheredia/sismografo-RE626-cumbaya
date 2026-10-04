"""Fase 3: baseline causal (migracion de filtfilt usuario) vs original.
Cadena A causal: demean -> /sens -> HP 0.7Hz SOS -> notch 29.79/18.07/42.24 Q=30 SOS -> BP 1-25Hz SOS.
Todo con scipy.signal.sosfilt (causal, streaming-compatible).
Original usuario (solo offline): iirnotch + filtfilt.
Metricas por evento: RMS ruido, SNR dB, CC vs limpia R53D1, latencia ms.
Uso: python scripts/04_baseline_causal.py
"""
import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy import read
from scipy.signal import butter, sosfilt, iirnotch, tf2sos, filtfilt
from scipy.signal import welch

SENS = 399650000.0  # counts/(m/s) RE626
FS = 100.0
NOTCHES = [(29.79, 30.0), (18.07, 30.0), (42.24, 30.0)]
FIGD = "results/figs"
METD = "results/metrics"
os.makedirs(FIGD, exist_ok=True)
os.makedirs(METD, exist_ok=True)

# SOS fijos (disenados una vez, reutilizados = eficiente edge)
SOS_HP = butter(2, 0.7, btype="highpass", fs=FS, output="sos")
SOS_BP = butter(4, [1.0, 25.0], btype="bandpass", fs=FS, output="sos")
SOS_NOTCH = []
for f0, Q in NOTCHES:
    b, a = iirnotch(w0=f0, Q=Q, fs=FS)
    SOS_NOTCH.append(tf2sos(b, a))

def causal_chain(counts):
    v = (counts.astype(float) - np.mean(counts)) / SENS
    y = sosfilt(SOS_HP, v)
    for s in SOS_NOTCH:
        y = sosfilt(s, y)
    y = sosfilt(SOS_BP, y)
    return y

def user_original(counts, fs=FS):
    x = counts.astype(float).copy()
    x = x - np.mean(x)
    for f0, Q in [(29.79, 30.0), (18.07, 30.0)]:
        b, a = iirnotch(w0=f0, Q=Q, fs=fs)
        x = filtfilt(b, a, x)
    return x / SENS  # sin remove_response completa, escala simple para comparar

def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))

def snr_db(sig, noise):
    return 20 * np.log10((rms(sig) + 1e-18) / (rms(noise) + 1e-18))

def cc(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0

EVENTS = ["ev_local_close_M45_20260915", "ev_local_M53_20260910"]

def process(tag):
    print(f"== {tag} ==")
    re626 = read(f"data/raw_mseed/{tag}_RE626.mseed")[0]
    ref = read(f"data/raw_mseed/{tag}_R53D1.mseed")[0]
    # alinear: recortar a interseccion temporal, resample si hace falta (ambos 100Hz)
    t0 = max(re626.stats.starttime, ref.stats.starttime)
    t1 = min(re626.stats.endtime, ref.stats.endtime)
    re626.trim(t0, t1); ref.trim(t0, t1)
    n = min(len(re626.data), len(ref.data))
    raw = re626.data[:n].astype(float)
    clean = ref.data[:n].astype(float)
    # ventanas: ruido 0-200s, senal 290-320s (origen al centro 300s)
    fs = 100
    noise_raw = raw[:200 * fs]
    sig_raw = raw[290 * fs:320 * fs]
    clean_sig = clean[290 * fs:320 * fs]
    # latencia cadena causal (600s)
    t = time.perf_counter()
    filt = causal_chain(raw)
    lat_ms = (time.perf_counter() - t) * 1000
    orig = user_original(raw)
    noise_f = filt[:200 * fs]
    sig_f = filt[290 * fs:320 * fs]
    # ref filtrada igual para CC justo
    ref_f = causal_chain(clean)
    ref_sig_f = ref_f[290 * fs:320 * fs]
    m = {
        "npts": n,
        "rms_noise_raw_counts": rms(noise_raw),
        "rms_noise_filt_m_s": rms(noise_f),
        "snr_raw_dB": snr_db(sig_raw, noise_raw),
        "snr_causal_dB": snr_db(sig_f, noise_f),
        "snr_orig_filtfilt_dB": snr_db(orig[290*fs:320*fs], orig[:200*fs]),
        "cc_raw_vs_ref": cc(sig_raw, clean_sig),
        "cc_causal_vs_refF": cc(sig_f, ref_sig_f),
        "latency_causal_600s_ms": lat_ms,
        "latency_per_60s_ms": lat_ms / 10,
    }
    print(f"   SNR raw={m['snr_raw_dB']:.1f} -> causal={m['snr_causal_dB']:.1f} (filtfilt={m['snr_orig_filtfilt_dB']:.1f})")
    print(f"   CC raw={m['cc_raw_vs_ref']:.2f} -> causal={m['cc_causal_vs_refF']:.2f} | lat {lat_ms:.0f}ms/600s ({m['latency_per_60s_ms']:.0f}ms/60s)")
    # fig comparativa
    tt = np.arange(n) / fs - 300  # 0 = origen aprox
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(tt, raw / SENS * 1e6, lw=0.4)
    ax[0].set_title(f"{tag} RE626 raw vel (um/s)")
    ax[0].axvline(0, color="r", alpha=0.5)
    ax[1].plot(tt, filt * 1e6, lw=0.4, label="causal SOS")
    ax[1].plot(tt, orig * 1e6, lw=0.3, alpha=0.6, label="orig filtfilt")
    ax[1].legend(fontsize=8); ax[1].axvline(0, color="r", alpha=0.5)
    ax[1].set_title("filtrado causal vs filtfilt (um/s)")
    ax[2].plot(tt, ref_f * 1e6, lw=0.4, color="g")
    ax[2].set_title("R53D1 ref causal (um/s)"); ax[2].axvline(0, color="r", alpha=0.5)
    ax[2].set_xlim(-300, 300); ax[2].set_xlabel("s rel origen")
    fig.tight_layout()
    fig.savefig(f"{FIGD}/{tag}_baseline.png", dpi=110)
    plt.close(fig)
    # PSD antes/despues (ventana ruido)
    f1, p1 = welch((raw[:200*fs] - raw[:200*fs].mean()) / SENS, fs=fs, nperseg=4096)
    f2, p2 = welch(noise_f, fs=fs, nperseg=4096)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.loglog(f1, p1, lw=0.8, label="raw"); ax2.loglog(f2, p2, lw=0.8, label="causal")
    for f0, _ in NOTCHES:
        ax2.axvline(f0, color="r", alpha=0.4, lw=0.8)
    ax2.set_xlim(0.5, 49); ax2.legend(); ax2.set_title(f"PSD ruido {tag}"); ax2.grid(True, alpha=0.3)
    fig2.tight_layout(); fig2.savefig(f"{FIGD}/{tag}_psd_after.png", dpi=110); plt.close(fig2)
    return m

def main():
    out = {tag: process(tag) for tag in EVENTS}
    json.dump(out, open(f"{METD}/baseline_causal.json", "w"), indent=2)
    print("OK", f"{METD}/baseline_causal.json")

if __name__ == "__main__":
    main()
