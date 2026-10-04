# Seismograph RE626 – Cumbayá, Ecuador

> Real-time seismic detector for noisy urban station `AM.RE626.00.EHZ` (100 Hz) cleaned by comparison with two quiet neighbor stations (`R2A1F`, `R53D1`).
> Causal prefilter → per-station STA/LTA → multi-station coincidence veto → PhaseNet picker on clean station.

## Key results

From `results/metrics/BENCHMARK.md` (2026-09-22) and `HOUR1.md`:

* **Noise map:** Day 2.3–2.7x noisier than night. Lines at 29.8 / 18.1 / 42.2 / 7.6 / 14.5 / 10.7 Hz. 60 Hz ~0 → no 60 Hz notch. RE626 8–10x noisier than neighbors.
* **Mono-station ceiling (proven):** No single-station filter alone passes. HP0.7+notch preserves P; aggressive BP, Hampel despike, causal gating, wavelet db4 (erases quake −7 dB), and EQTransformer-1C all rejected with evidence.
* **Multi-station veto (PASS):** 73–75% false-alarm rejection, 0 false/hour at night, burst vetoed, S-phase kept.
* **PhaseNet picker (system PASS / mono FAIL):** Original PhaseNet on CPU, 1C-triplicated input. RE626 P 0.035 (no pick), R2A1F P 0.173 (pick at thr 0.15). Burst 0.013 / night 0.006 (0 false).
* **1-hour noisy test:** 76 raw triggers → 25 after veto → **4 final alerts** (possible Quito M<3 microseismicity, pending analyst review).
* **Edge budget:** 0.4 ms pre + 106 ms STA/LTA per 60 s + 1.5 ms/s PhaseNet on 1 thread (~500x real-time). Runs on any PC, no GPU, no Docker.

Production pipeline: **F3-light → STA/LTA per station → coincidence veto → PhaseNet on kept (pick on clean)**

## How it works

1. **Prefilter (causal):** HP 0.7 Hz (order 2) + notches 29.79 / 18.07 / 42.24 Hz (Q=30). No aggressive BP.
2. **Detection:** Band 2–20 Hz, STA 1.0 s / LTA 20.0 s. Thresholds on/off: RE626 3.2/1.4, R2A1F 2.8/1.2, R53D1 3.5/1.5.
3. **Veto:** Keep RE626 trigger only if a clean station triggers within onset tol 5 s with overlap pad 2 s.
4. **Picker:** PhaseNet (seisbench original) thr P 0.15 on clean station.

See `config.yaml` — this is the signed production decision.

## Repo layout

* `config.yaml` – production parameters
* `requirements.txt` – obspy, numpy, scipy, matplotlib, pyyaml, PyWavelets, torch (CPU), seisbench
* `scripts/` – in execution order: `00_verify_access.py`, `01_download_windows.py`, `02_noise_map.py`, `03_download_events.py`, `04_baseline_causal.py`, `04b_baseline_loop.py`, `04c_techo_mono.py`, `04d_cadena_v2.py`, `05_veto.py`, `06_fase4.py`, `07_veto_prod.py`, `08_fase6_picker.py`, `09_cascada.py`, `pipeline.py`, `run_cluster.py`, `rs_fdsn.py`
* `results/metrics/` – JSON + gates (`G1`, `G3`, `G4`, `G5_final`, `G6`) + `BENCHMARK.md` + `HOUR1.md` + `FASE8.md`
* `results/figs/` – PSD, spectrograms, veto plots
* `entrega/` – standalone usable version: `detector.py`, `vivo.py`, `config.yaml`, `datos_ejemplo/RE626_test.mseed`, `ejemplo.bat`, `vivo.bat`
* `data/inventories/` – station metadata only
* `SESION.md` – session log (Spanish)

## Installation

```bash
pip install -r requirements.txt
# torch CPU recommended via https://download.pytorch.org/whl/cpu
pip install torch seisbench
Usage
Quick test (no Shake needed):
python entrega/detector.py --mseed entrega/datos_ejemplo/RE626_test.mseed
With veto (real cleaning):
python entrega/detector.py --mseed RE626.mseed --ref1 R2A1F.mseed --ref2 R53D1.mseed
With AI picker:
python entrega/detector.py --mseed RE626.mseed --ref1 R2A1F.mseed --ref2 R53D1.mseed --phasenet
Research pipeline:
python scripts/pipeline.py --mseed entrega/datos_ejemplo/RE626_test.mseed
python scripts/run_cluster.py
Live (PC on same network as Shake):
python entrega/vivo.py --simular   # drill without hardware
python entrega/vivo.py --host rs.local  # or --host 192.168.1.50 --puerto 18000
Rule: without --ref1/--ref2 there is no cleaning (raw triggers only).
Data
Large .mseed files are not in this repo (GitHub 100 MB limit). Included sample only: entrega/datos_ejemplo/RE626_test.mseed.
To reproduce full dataset (Raspberry Shake FDSN, cert expired → verify=False workaround in scripts/rs_fdsn.py, ~1 h archive delay):
python scripts/00_verify_access.py
python scripts/01_download_windows.py
python scripts/03_download_events.py
Benchmark events: M4.5 close 2026-09-15 (~70 km), M5.3 2026-09-10 (~400 km), regional, teleseismic M6.7.
Limitations / Roadmap
1. Review 4 final alerts in DataView (01:08, 01:50, 01:56, 02:03 UTC 2026-09-22).
2. Live access to Shake (IP/UDP or continued 1 h-delayed FDSN).
3. 24/7 edge PC + 2–3 day test.
4. PhaseNet-1C fine-tune (needs 20+ quakes + GPU/Colab).
License
MIT – see LICENSE.

**GitHub repo description (About field):**
> Seismic detector for noisy station RE626 (Cumbayá): causal filter + multi-station veto + PhaseNet. 73-75% fewer false alarms, 500x real-time.
