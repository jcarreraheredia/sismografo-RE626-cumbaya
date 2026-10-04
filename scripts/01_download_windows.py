"""Fase 0 G0 + Fase 1 descarga: 3 ventanas RE626 + inventario + estaciones cercanas.
Ventanas (America/Guayaquil UTC-5):
 - pico parqueadero: 2026-09-19T18:00 Quito = 23:00Z
 - noche quieta:     2026-09-20T03:00 Quito = 08:00Z
 - lab:              2026-09-19T10:00 Quito = 15:00Z
Cada una 10 min para mapa de ruido (suficiente para PSD, liviano).
Uso: python scripts/01_download_windows.py
"""
import os
from rs_fdsn import dataselect, station_text, station_xml

OUT = "data/raw_mseed"
INV = "data/inventories"
os.makedirs(OUT, exist_ok=True)
os.makedirs(INV, exist_ok=True)

WINDOWS = {
    "pico": ("2026-09-19T23:00:00", "2026-09-19T23:10:00"),
    "noche": ("2026-09-20T08:00:00", "2026-09-20T08:10:00"),
    "lab": ("2026-09-19T15:00:00", "2026-09-19T15:10:00"),
}

def main():
    print("== Descarga RE626 ==")
    for name, (t0, t1) in WINDOWS.items():
        fp = f"{OUT}/RE626_{name}_10min.mseed"
        if os.path.exists(fp) and os.path.getsize(fp) > 1000:
            print(f"  {name}: existe {fp}, skip")
            continue
        data, url = dataselect("AM", "RE626", "00", "EHZ", t0, t1)
        open(fp, "wb").write(data)
        print(f"  {name}: {len(data)} bytes -> {fp}")

    print("\n== Inventario RE626 ==")
    try:
        xml, url = station_xml("AM", "RE626")
        open(f"{INV}/RE626_response.xml", "wb").write(xml)
        print(f"  OK {len(xml)} bytes")
    except Exception as e:
        print(f"  FAIL inventario: {e}")

    print("\n== Estaciones cercanas Cumbaya (-0.198,-78.433) radio 0.5deg ==")
    q = "net=AM&level=station&minlatitude=-0.7&maxlatitude=0.3&minlongitude=-78.93&maxlongitude=-77.93"
    try:
        txt = station_text(q)
        open("data/inventories/AM_cercanas.txt", "w").write(txt)
        lines = [l for l in txt.strip().split("\n") if not l.startswith("#") and l.strip()]
        print(f"  {len(lines)} estaciones AM en radio:")
        for l in lines[:30]:
            print("   ", l)
    except Exception as e:
        print(f"  FAIL cercanas: {e}")

    print("\n== Test regresion ANMO (EARTHSCOPE, no RS) ==")
    try:
        from obspy import UTCDateTime
        from obspy.clients.fdsn import Client
        c = Client("EARTHSCOPE")
        t = UTCDateTime("2026-08-20T14:00:00")
        st = c.get_waveforms("IU", "ANMO", "00", "BHZ", t, t + 120)
        print(f"  ANMO OK: {st[0].id} sr={st[0].stats.sampling_rate}")
    except Exception as e:
        print(f"  ANMO FAIL (no bloquea G0 RS): {str(e)[:200]}")

if __name__ == "__main__":
    main()
