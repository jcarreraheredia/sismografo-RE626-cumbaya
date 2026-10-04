"""Cascada final: PhaseNet en ventana limpia coincidente para cada kept del veto.
Uso: python scripts/09_cascada.py --tag hour1
"""
import argparse, json
import numpy as np
from obspy import read, Stream, Trace, UTCDateTime

SENS = 399650000.0
FS = 100.0

def to_3c(tr, t_center, half=15.0):
    c = tr.copy().trim(t_center - half, t_center + half)
    if len(c.data) < 100:
        return None, None
    c.detrend("demean")
    c.filter("bandpass", freqmin=1.0, freqmax=45.0, corners=4, zerophase=True)
    d = c.data.astype(float) / SENS
    d = d / (np.abs(d).max() + 1e-12)
    st = Stream()
    for ch in ["EHZ", "EHN", "EHE"]:
        t = Trace(data=d.astype(np.float32))
        t.stats.update({"network": "AM", "station": tr.stats.station,
                        "location": "00", "channel": ch,
                        "sampling_rate": FS, "starttime": c.stats.starttime})
        st.append(t)
    return st, c.stats.starttime

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hour1")
    a = ap.parse_args()
    import seisbench.models as sbm
    model = sbm.PhaseNet.from_pretrained("original")
    model.eval()
    cl = json.load(open(f"results/metrics/cluster_{a.tag}.json"))
    # OJO: cluster_hour1 se genero con umbrales viejos; recalcula kept con nuevos via run_cluster tras actualizar config
    trs = {s: read(f"data/raw_mseed/{a.tag}_{s}.mseed")[0] for s in ["RE626", "R2A1F", "R53D1"]}
    surv = []
    for iso in cl["kept"]:
        t = UTCDateTime(iso)
        # elige limpia con trigger mas cercano (usa R53D1 por defecto)
        best, bestdt = "R53D1", 1e9
        for s in ["R2A1F", "R53D1"]:
            pass
        st3, _ = to_3c(trs["R53D1"], t)
        if st3 is None:
            continue
        pred = model.annotate(st3)
        mx = {p.stats.channel: float(np.asarray(p.data).max()) for p in pred}
        keep = mx.get("PhaseNet_P", 0) >= 0.15 or mx.get("PhaseNet_S", 0) >= 0.15
        surv.append((iso, round(mx.get("PhaseNet_P", 0), 3), round(mx.get("PhaseNet_S", 0), 3), keep))
        print(iso, mx, "SURVIVE" if keep else "drop")
    print(f"\n{a.tag}: {len(surv)} kept veto -> {sum(s[3] for s in surv)} sobreviven PhaseNet")
    json.dump(surv, open(f"results/metrics/cascada_{a.tag}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
