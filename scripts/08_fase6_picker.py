"""Fase 6: PhaseNet (SeisBench, CPU) sobre ventanas kept para recuperar P perdida.
Dato 1C EHZ -> se triplica a 3C (workaround documentado; S menos fiable, P objetivo).
Ventanas: cada kept +-30s + noche 60s (falsos) + P catalogo +-15s (test recuperacion).
Uso: python scripts/08_fase6_picker.py
"""
import json, time
import numpy as np
from obspy import read, Stream, Trace, UTCDateTime

SENS = 399650000.0
FS = 100.0

def to_3c(tr, sec0, sec1):
    c = tr.copy().trim(tr.stats.starttime + sec0, tr.stats.starttime + sec1)
    c.detrend("demean")
    c.filter("bandpass", freqmin=1.0, freqmax=45.0, corners=4, zerophase=True)
    d = (c.data.astype(float) / SENS)
    d = d / (np.abs(d).max() + 1e-12)  # norm ventana
    st = Stream()
    for ch in ["EHZ", "EHN", "EHE"]:
        t = Trace(data=d.astype(np.float32))
        t.stats.update({"network": "AM", "station": tr.stats.station,
                        "location": "00", "channel": ch,
                        "sampling_rate": FS, "starttime": c.stats.starttime})
        st.append(t)
    return st, c.stats.starttime

def main():
    import seisbench.models as sbm
    print("cargando PhaseNet original...")
    model = sbm.PhaseNet.from_pretrained("original")
    model.eval()
    print("OK", type(model).__name__)

    import inspect
    print("annotate" in dir(model), "classify" in dir(model), "pick_onset" in dir(model))

    tests = []
    # 1. P catalogo ev_close (STA la perdio en RE626): debe recuperarla
    tr = read("data/raw_mseed/ev_local_close_M45_20260915_RE626.mseed")[0]
    t0 = tr.stats.starttime
    pcat = float(UTCDateTime("2026-09-15T23:45:00") - t0)
    tests.append(("RE626_P_perdida", tr, max(pcat - 15, 0), pcat + 15))
    # 2. kept S (control positivo)
    tests.append(("RE626_S_kept", tr, 290 - 30, 290 + 60))
    # 3. burst vetado (control negativo: no debe dar P alta)
    tests.append(("RE626_burst", tr, 240 - 15, 240 + 30))
    # 4. noche 60s (falsos)
    trn = read("data/raw_mseed/RE626_noche_10min.mseed")[0]
    tests.append(("RE626_noche", trn, 200, 260))
    # 5. limpia R2A1F P (control positivo limpio)
    tr2 = read("data/raw_mseed/ev_local_close_M45_20260915_R2A1F.mseed")[0]
    p2 = float(UTCDateTime("2026-09-15T23:45:00") - tr2.stats.starttime)
    tests.append(("R2A1F_P", tr2, max(p2 - 15, 0), p2 + 15))

    res = {}
    for name, trace, a, b in tests:
        st3, stime = to_3c(trace, a, b)
        t = time.perf_counter()
        try:
            pred = model.annotate(st3)
        except Exception as e:
            res[name] = {"error": str(e)[:200]}
            print(name, "ERROR", str(e)[:150])
            continue
        lat = (time.perf_counter() - t) * 1000
        # pred: Stream con traces P,S,noise? extrae maximos
        info = {"lat_ms": round(lat, 1), "win_s": round(b - a, 1)}
        for p in pred:
            d = np.asarray(p.data, float)
            info[f"{p.stats.channel}_max"] = round(float(d.max()), 3)
            info[f"{p.stats.channel}_tmax_s"] = round(float(np.argmax(d) / FS), 2)
        res[name] = info
        print(name, json.dumps(info))
    json.dump(res, open("results/metrics/fase6_phasenet.json", "w"), indent=2)
    print("OK results/metrics/fase6_phasenet.json")

if __name__ == "__main__":
    main()
