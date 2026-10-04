"""Fase 1 G1: mapa de ruido RE626 - PSD Welch + espectrograma + RMS + deteccion de lineas.
Loop: si no se confirman lineas, no se disena notch.
Uso: python scripts/02_noise_map.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy import read
from scipy.signal import welch, spectrogram, find_peaks

FILES = {
    "pico": "data/raw_mseed/RE626_pico_10min.mseed",
    "noche": "data/raw_mseed/RE626_noche_10min.mseed",
    "lab": "data/raw_mseed/RE626_lab_10min.mseed",
}
FIGD = "results/figs"
METD = "results/metrics"
os.makedirs(FIGD, exist_ok=True)
os.makedirs(METD, exist_ok=True)

def analyze(name, fp):
    st = read(fp)
    tr = st[0]
    fs = float(tr.stats.sampling_rate)
    x = tr.data.astype(float)
    x = x - np.mean(x)
    rms = float(np.sqrt(np.mean(x ** 2)))
    std = float(np.std(x))
    # PSD Welch: nperseg 4096 -> resol ~0.024Hz, bueno para lineas
    f, p = welch(x, fs=fs, nperseg=4096, noverlap=2048)
    mask = (f >= 0.5) & (f <= 49)
    f, p = f[mask], p[mask]
    # picos: prominencia relativa al fondo (percentil 90 en ventana)
    logp = 10 * np.log10(p + 1e-30)
    peaks, props = find_peaks(logp, prominence=8, distance=8)
    lines = [(float(f[i]), float(logp[i])) for i in peaks if f[i] > 1.0]
    lines = sorted(lines, key=lambda t: -t[1])[:12]
    # bandas de energia
    def band(a, b):
        m = (f >= a) & (f < b)
        return float(np.trapezoid(p[m], f[m])) if np.any(m) else 0.0
    e_traffic = band(1, 8)
    e_build = band(8, 15)
    e_mach = band(15, 30)
    e_elec = band(55, 65)
    e_tot = band(0.5, 49)
    # espectrograma
    ff, tt, S = spectrogram(x, fs=fs, nperseg=1024, noverlap=768)
    # figs
    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    ax[0].plot(tr.times(), x, lw=0.4)
    ax[0].set_title(f"{tr.id} {name} counts-demeaned RMS={rms:.0f}")
    ax[0].set_xlabel("s")
    pcm = ax[1].pcolormesh(tt, ff, 10 * np.log10(S + 1e-30), shading="gouraud", vmin=np.percentile(10*np.log10(S+1e-30), 5), vmax=np.percentile(10*np.log10(S+1e-30), 98))
    ax[1].set_ylim(0, 50)
    ax[1].set_ylabel("Hz")
    ax[1].set_xlabel("s")
    fig.colorbar(pcm, ax=ax[1], label="dB")
    fig.tight_layout()
    fig.savefig(f"{FIGD}/RE626_{name}_wave_spec.png", dpi=120)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.semilogy(f, p)
    for fl, vl in lines:
        ax2.axvline(fl, color="r", alpha=0.4, lw=0.8)
        ax2.text(fl, p[np.argmin(abs(f - fl))] * 1.5, f"{fl:.2f}Hz", rotation=90, fontsize=7, color="r")
    ax2.set_title(f"PSD Welch RE626 {name} fs={fs}Hz")
    ax2.set_xlabel("Hz")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"{FIGD}/RE626_{name}_psd.png", dpi=120)
    plt.close(fig2)

    return {
        "file": fp, "fs": fs, "npts": int(tr.stats.npts),
        "rms_counts": rms, "std_counts": std,
        "E_1_8_traffic": e_traffic, "E_8_15_build": e_build,
        "E_15_30_machines": e_mach, "E_55_65_elec": e_elec, "E_tot": e_tot,
        "frac_traffic": e_traffic / e_tot, "frac_machines": e_mach / e_tot,
        "lines_Hz_dB": lines,
    }

def main():
    out = {}
    for name, fp in FILES.items():
        print(f"-- {name}: {fp}")
        out[name] = analyze(name, fp)
        print(f"   RMS={out[name]['rms_counts']:.0f} lineas={[(round(a,2)) for a,_ in out[name]['lines_Hz_dB'][:6]]}")
    json.dump(out, open(f"{METD}/RE626_noise_map.json", "w"), indent=2)
    # reporte
    rep = ["# Mapa de ruido RE626 (G1)\n"]
    for k, v in out.items():
        rep.append(f"## {k}: RMS={v['rms_counts']:.0f} counts, fs={v['fs']}Hz")
        rep.append(f"- fraccion energia 1-8Hz (trafico/edificio): {v['frac_traffic']:.2f}")
        rep.append(f"- fraccion 15-30Hz (maquinas): {v['frac_machines']:.2f}")
        rep.append(f"- lineas detectadas: {', '.join(f'{a:.2f}Hz' for a,_ in v['lines_Hz_dB'])}")
        rep.append(f"- figs: results/figs/RE626_{k}_wave_spec.png, RE626_{k}_psd.png\n")
    open(f"{METD}/RE626_noise_map.md", "w").write("\n".join(rep))
    print("\n".join(rep))
    print("OK metrics + figs")

if __name__ == "__main__":
    main()
