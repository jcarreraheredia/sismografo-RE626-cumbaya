# Bitácora de sesión — Proyecto Sismógrafo RE626

## Resumen en cristiano (por fase)
- **F0**: entorno listo; acceso FDSN con workaround por certificado expirado. RE626 existe (ago-2026, Cumbayá).
- **F1**: mapa de ruido (pico/noche/lab). Día 2.5x más ruidoso; líneas 29.8/18.1/42.2/7.6/14.5/10.7 Hz; 60Hz ~0. RE626 10x más ruidoso que vecinas.
- **F2**: dataset benchmark (M4.5 cercano, M5.3, regional, tele M6.7) + limpias R2A1F/R53D1. R0949 muerta.
- **F3**: filtro causal (HP0.7+notch, sin BP agresivo). Techo mono demostrado; despike descartado.
- **F4**: gating causal y wavelet RECHAZADOS (wavelet borra el sismo −7dB).
- **F5**: veto multi-estación PASS (73-75% rechazo, 0 falsos/hora de noche, burst vetado).
- **F6**: PhaseNet PASS sistema / FAIL mono. Pick en limpia, RE626 confirma.
- **F7**: BENCHMARK.md firmado. **F8**: pipeline + config + prueba 1h (76→25→4 alertas).

## Mapa de archivos
- `config.yaml`: decisión de producción (umbrales 3.2/2.8/3.5, tol 5s, pad 2s, thr PhaseNet 0.15).
- `scripts/00→09, pipeline.py, run_cluster.py`: en orden de ejecución.
- `results/metrics/`: JSON + gates (G1,G3,G4,G5_final,G6) + BENCHMARK.md + HOUR1.md + FASE8.md.
- `results/figs/`: PSD, espectrogramas, comparativas, vetos.
- `requirements.txt`: dependencias (torch CPU + seisbench incluidos).

## Pendientes
1. Revisar 4 alertas en DataView (01:08, 01:50, 01:56, 02:03 UTC 22-sep-2026).
2. Acceso vivo al Shake (IP/UDP o seguir FDSN con retardo 1h).
3. Definir edge (posible Dockerfile entonces).
4. Fine-tune PhaseNet 1C (necesita 20+ sismos + GPU/Colab).

## Notas operativas
- FDSN RS: cert expirado → `ssl._create_unverified_context()` (ver `rs_fdsn.py`).
- Retardo archivo FDSN ~1h (404 en ventana reciente).
- Bug YAML: claves `on/off` siempre entrecomilladas.
