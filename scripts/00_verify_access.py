"""Fase 0 G0: verifica acceso FDSN RASPISHAKE + regresion ANMO. Offline-first."""
import sys
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

def test_raspishake():
    print("== Test RASPISHAKE RE626 ==")
    rs = Client("RASPISHAKE")
    # 1. metadata
    try:
        inv = rs.get_stations(network="AM", station="RE626", level="station")
        print(f"Metadata OK: {inv}")
        for net in inv:
            for sta in net:
                print(f"  Station {net.code}.{sta.code} lat={sta.latitude} lon={sta.longitude} elev={sta.elevation}")
                for ch in sta.channels if hasattr(sta, 'channels') else []:
                    print(f"    {ch.location_code}.{ch.code} sr={ch.sample_rate}")
    except Exception as e:
        print(f"Metadata FAIL: {e}")
        return False
    # 2. waveform reciente: prueba ultimos dias
    for d in ["2026-09-20T12:00:00", "2026-09-15T12:00:00", "2026-08-20T14:00:00", "2026-09-21T03:00:00"]:
        t = UTCDateTime(d)
        try:
            st = rs.get_waveforms("AM", "RE626", "00", "EHZ", t, t + 120)
            tr = st[0]
            print(f"Waveform OK {d}: {tr.id} sr={tr.stats.sampling_rate} npts={tr.stats.npts} gaps={len(st)}")
            st.write("data/raw_mseed/RE626_probe.mseed", format="MSEED")
            print("  Guardado data/raw_mseed/RE626_probe.mseed")
            return True
        except Exception as e:
            print(f"  {d} FAIL: {str(e)[:200]}")
    return False

def test_anmo_regression():
    print("\n== Test regresion IU.ANMO (codigo usuario) ==")
    try:
        c = Client("EARTHSCOPE")
        t = UTCDateTime("2026-08-20T14:00:00")
        st = c.get_waveforms("IU", "ANMO", "00", "BHZ", t, t + 120)
        print(f"ANMO OK: {st[0].id} sr={st[0].stats.sampling_rate} npts={st[0].stats.npts}")
        return True
    except Exception as e:
        print(f"ANMO FAIL: {str(e)[:300]}")
        return False

if __name__ == "__main__":
    ok_rs = test_raspishake()
    ok_anmo = test_anmo_regression()
    print(f"\nG0: RASPISHAKE={ok_rs} ANMO={ok_anmo}")
    sys.exit(0 if ok_rs else 1)
