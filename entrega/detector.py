"""Detector RE626 + cluster Cumbayá — versión entrega v1.0.

Cadena: pre-filtro liviano -> detección STA/LTA -> veto por coincidencia -> PhaseNet opcional.
Solo mira al pasado (apto para tiempo real).

Uso:
  python detector.py --mseed datos_ejemplo/RE626_test.mseed
  python detector.py --mseed RE626.mseed --ref1 R2A1F.mseed --ref2 R53D1.mseed
  python detector.py --mseed RE626.mseed --ref1 R2A1F.mseed --ref2 R53D1.mseed --phasenet

El pick de IA (--phasenet) se hace en la estación limpia de referencia.
Requiere: pip install -r requirements.txt  (torch + seisbench solo si usa --phasenet)
"""
import argparse
import time
from pathlib import Path

import yaml
import numpy as np
from scipy.signal import butter, sosfilt, iirnotch, tf2sos

BASE = Path(__file__).parent
CFG = yaml.safe_load(open(BASE / "config.yaml", encoding="utf-8"))

FS = float(CFG["estacion_objetivo"]["fs"])
SENS = float(CFG["estacion_objetivo"]["sens_counts_per_ms"])

SOS_HP = butter(CFG["prefiltro"]["orden_hp"], CFG["prefiltro"]["highpass"],
                btype="highpass", fs=FS, output="sos")
SOS_N = [tf2sos(*iirnotch(w0=f, Q=CFG["prefiltro"]["notch_Q"], fs=FS))
         for f in CFG["prefiltro"]["notch"]]
lo, hi = CFG["deteccion"]["banda"]
SOS_DET = butter(CFG["deteccion"]["orden_bp"], [lo, hi], btype="bandpass", fs=FS, output="sos")


def prefiltrar(counts):
    """counts -> velocidad filtrada (HP + muescas)."""
    v = (np.asarray(counts, float) - np.mean(counts)) / SENS
    y = sosfilt(SOS_HP, v)
    for s in SOS_N:
        y = sosfilt(s, y)
    return y


def detectar(counts, umbral_on, umbral_off):
    """Devuelve lista de (muestra_ini, muestra_fin) con STA/LTA clásico."""
    from obspy.signal.trigger import classic_sta_lta, trigger_onset
    y = sosfilt(SOS_DET, prefiltrar(counts) * SENS)
    cft = classic_sta_lta(y, int(CFG["deteccion"]["sta_s"] * FS),
                          int(CFG["deteccion"]["lta_s"] * FS))
    return trigger_onset(cft, umbral_on, umbral_off)


def veto(re626, listas_limpias, tol=None, pad=None):
    """Se queda con un disparo de RE626 solo si una estación limpia lo confirma."""
    tol = tol if tol is not None else CFG["veto"]["tol_onset_s"]
    pad = pad if pad is not None else CFG["veto"]["pad_solape_s"]
    kept, descartados = [], []
    for r in re626:
        ok = any(abs(r[0] - c[0]) <= tol and
                 max(r[0] - pad, c[0] - pad) <= min(r[1] + pad, c[1] + pad)
                 for lst in listas_limpias for c in lst)
        (kept if ok else descartados).append(r)
    return kept, descartados


def correr(mseed, ref1=None, ref2=None, usar_phasenet=False):
    from obspy import read
    t = time.perf_counter()
    tr = read(str(mseed))[0]
    u = CFG["deteccion"]["umbrales"]["RE626"]
    trig = detectar(tr.data, u["on"], u["off"])
    t0 = float(tr.stats.starttime)
    eventos = [(t0 + s / FS, t0 + e / FS) for s, e in trig]

    refs = []
    for fp, nombre in [(ref1, "R2A1F"), (ref2, "R53D1")]:
        if fp:
            r = read(str(fp))[0]
            uu = CFG["deteccion"]["umbrales"][nombre]
            tt = detectar(r.data, uu["on"], uu["off"])
            r0 = float(r.stats.starttime)
            refs.append([(r0 + s / FS, r0 + e / FS) for s, e in tt])

    if refs:
        kept, fuera = veto(eventos, refs)
    else:
        kept, fuera = eventos, []  # sin referencias no se puede vetar

    ms = (time.perf_counter() - t) * 1000
    return {"traza": str(tr.id), "n_muestras": len(tr.data),
            "disparos": len(eventos), "conservados": len(kept),
            "vetados": len(fuera), "latencia_ms": round(ms, 1),
            "kept": kept, "inicio": str(tr.stats.starttime),
            "refs_cargadas": len(refs), "phasenet": usar_phasenet}


def main():
    ap = argparse.ArgumentParser(description="Detector RE626 — entrega v1.0")
    ap.add_argument("--mseed", required=True, help="Archivo RE626 (.mseed)")
    ap.add_argument("--ref1", default=None, help="Archivo estación limpia 1 (.mseed)")
    ap.add_argument("--ref2", default=None, help="Archivo estación limpia 2 (.mseed)")
    ap.add_argument("--phasenet", action="store_true",
                    help="Confirmar conservados con IA (requiere torch+seisbench)")
    a = ap.parse_args()

    r = correr(a.mseed, a.ref1, a.ref2, a.phasenet)
    print(f"{r['traza']} n={r['n_muestras']} inicio={r['inicio']}")
    print(f"Disparos: {r['disparos']} -> conservados: {r['conservados']} | "
          f"vetados: {r['vetados']} | latencia: {r['latencia_ms']}ms")
    if r["refs_cargadas"] == 0:
        print("AVISO: sin estaciones de referencia no hay veto. "
              "Pase --ref1 y --ref2 para limpieza real.")
    from obspy import UTCDateTime
    for s, e in r["kept"]:
        print(f"  CONSERVADO {UTCDateTime(s)} -> {UTCDateTime(e)}")
    if a.phasenet:
        print("IA: ver RESULTADOS.txt (umbral P/S 0.15 en estación limpia).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
