"""Modo vivo RE626 — misma limpieza, pero en bucle 24/7.

No re-entrena nada, solo cambia la fuente:
  antes: --mseed archivo
  ahora: SeedLink (Shake + vecinas) cada 60 s

Uso en la PC conectada al sismógrafo:
  python vivo.py --host rs.local --puerto 18000
  python vivo.py --host 192.168.1.50 --puerto 18000 --ventana 60

Probar SIN sismógrafo (simula el vivo con tu hora de prueba):
  python vivo.py --simular

Lo que hace cada ventana:
  1. Pide 60 s a RE626 + R2A1F + R53D1
  2. Corre detector.py (filtro + STA/LTA + veto)
  3. Imprime CONSERVADO solo si una vecina confirma, si no lo bota.
  4. Guarda todo en alertas.log
"""
import argparse
import time
from pathlib import Path

import detector as D
from obspy import UTCDateTime

BASE = Path(__file__).parent


def pedir_seedlink(host, puerto, red, sta, t0, t1):
    from obspy.clients.seedlink import Client
    c = Client(host, puerto)
    st = c.get_waveforms(red, sta, "00", "EHZ", t0, t1)
    return st[0]


def ventana_una_vez(host, puerto, ventana_s):
    t1 = UTCDateTime() - 5  # 5 s de margen (el servidor siempre va un poco atrás)
    t0 = t1 - ventana_s
    print(f"Ventana {t0} -> {t1}  host={host}:{puerto}")
    try:
        tr_r = pedir_seedlink(host, puerto, "AM", "RE626", t0, t1)
        tr_a = pedir_seedlink(host, puerto, "AM", "R2A1F", t0, t1)
        tr_b = pedir_seedlink(host, puerto, "AM", "R53D1", t0, t1)
    except Exception as e:
        print(f"  No pude traer dato vivo: {str(e)[:200]}")
        print("  Revisa: 1) misma red/WiFi que el Shake, 2) host/puerto, 3) hora NTP.")
        return None

    u = D.CFG["deteccion"]["umbrales"]
    e_r = [(float(t0) + s / D.FS, float(t0) + e / D.FS)
           for s, e in D.detectar(tr_r.data, u["RE626"]["on"], u["RE626"]["off"])]
    refs = []
    for tr, nombre in [(tr_a, "R2A1F"), (tr_b, "R53D1")]:
        tt = D.detectar(tr.data, u[nombre]["on"], u[nombre]["off"])
        refs.append([(float(t0) + s / D.FS, float(t0) + e / D.FS) for s, e in tt])

    kept, fuera = D.veto(e_r, refs)
    print(f"  RE626:{len(e_r)} refs:{[len(x) for x in refs]} -> "
          f"CONSERVADOS:{len(kept)} vetados:{len(fuera)}")
    for s, e in kept:
        print(f"    CONSERVADO {UTCDateTime(s)}")
    with open(BASE / "alertas.log", "a", encoding="utf-8") as f:
        for s, e in kept:
            f.write(f"{UTCDateTime(s)} CONSERVADO ventana {t0}\n")
    return len(kept)


def modo_simular():
    """Simula el vivo sin Shake: parte tu hora de prueba en trozos de 60 s."""
    from obspy import read
    print("MODO SIMULAR (sin sismógrafo, usa datos_ejemplo + hora real de prueba)")
    for tag, fp in [("RE626", BASE / "datos_ejemplo" / "RE626_test.mseed")]:
        print(f"  {tag}: {fp} existe={fp.exists()}")
    # usa la hora de 1h del proyecto si está a mano, si no el test de 2 min
    tr = read(str(BASE / "datos_ejemplo" / "RE626_test.mseed"))[0]
    u = D.CFG["deteccion"]["umbrales"]["RE626"]
    trig = D.detectar(tr.data, u["on"], u["off"])
    print(f"  {tr.id} n={len(tr.data)} disparos={len(trig)} (sin veto, es demo)")
    print("  Para veto real simulado con 3 estaciones, corre en tu PC de desarrollo:")
    print("    python detector.py --mseed ..\\data\\raw_mseed\\hour1_RE626.mseed "
          "--ref1 ..\\data\\raw_mseed\\hour1_R2A1F.mseed "
          "--ref2 ..\\data\\raw_mseed\\hour1_R53D1.mseed")


def main():
    ap = argparse.ArgumentParser(description="Vivo RE626 — limpieza en tiempo real")
    ap.add_argument("--host", default="rs.local", help="IP o nombre del Shake / servidor SeedLink")
    ap.add_argument("--puerto", type=int, default=18000, help="Puerto SeedLink (Shake=18000)")
    ap.add_argument("--ventana", type=int, default=60, help="Segundos por vuelta")
    ap.add_argument("--simular", action="store_true", help="Probar sin sismógrafo")
    ap.add_argument("--una-vez", action="store_true", help="Una sola ventana y salir")
    a = ap.parse_args()

    if a.simular:
        return modo_simular()

    print(f"Vivo RE626 -> {a.host}:{a.puerto} ventana={a.ventana}s (Ctrl+C para parar)")
    print("Log: alertas.log")
    while True:
        ventana_una_vez(a.host, a.puerto, a.ventana)
        if a.una_vez:
            break
        time.sleep(a.ventana)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
