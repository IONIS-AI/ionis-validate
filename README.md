# ionis-validate

Validation suite for the IONIS V20 HF propagation model. Run 62 physics
tests, predict SNR for any HF path, or validate the model against your
own QSO log — all from the command line or a browser UI, on any platform.

IONIS (Ionospheric Neural Inference System) predicts HF radio signal
strength from WSPR, RBN, and contest data. The V20 model was trained on
31 million propagation observations spanning 2005-2025.

## Install

```
pip install ionis-validate
```

Requires Python 3.9+ and PyTorch 2.0+. Works on Windows, macOS, and Linux.

## Quick Start

```bash
# Show model and system info
ionis-validate info

# Run the full 62-test validation suite
ionis-validate test

# Predict SNR for a single path
ionis-validate predict \
    --tx-grid FN20 --rx-grid IO91 --band 20m \
    --sfi 150 --kp 2 --hour 14 --month 6
```

## Browser UI

A point-and-click dashboard wrapping every command. Requires Python 3.10+.

```
pip install "ionis-validate[ui]"
ionis-validate ui
```

Opens a browser tab at `http://localhost:8765` with tabs for Predict,
Custom, ADIF, Report, and Info.

## Validate Your Log

Export your QSO log as an ADIF (.adi) file, then check how often the
model agrees the band was open for each contact. Both grids come from
the log itself (`MY_GRIDSQUARE` and `GRIDSQUARE`).

```bash
ionis-validate adif my_log.adi
```

QRZ exports work out of the box. LoTW requires both "Include QSL details"
and "Include QSO station details" checked on the download form.

All processing happens locally. Callsigns are stripped at parse time and
never leave your machine. The tool extracts only grid pairs, band, mode,
and time — no personal information.

## Batch Predictions

Define a set of paths in a JSON file and run them all at once:

```bash
ionis-validate custom my_paths.json
```

See the [Custom Path Tests](https://ionis-ai.com/testing/custom-paths/)
documentation for the JSON format.

## Beta Testing

If you are testing V20, follow the step-by-step
[Beta Test Plan](https://ionis-ai.com/testing/beta-test-plan/) (Test-1
through Test-9). It tells you exactly what to run, what to expect, and
how to submit your results.

Generate a structured report for filing as a GitHub Issue:

```bash
ionis-validate report
```

## Privacy

IONIS processes only grid-pair geometry, band, time, and solar indices.
No callsigns, names, or personal data are used by the model or stored
by this tool. ADIF log validation strips all PII at parse time.

Full privacy policy: <https://ionis-ai.com/about/data-privacy/>

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

## Links

- Documentation: <https://ionis-ai.com/testing/>
- Beta Test Plan: <https://ionis-ai.com/testing/beta-test-plan/>
- Source: <https://github.com/IONIS-AI/ionis-validate>
- Issues: <https://github.com/IONIS-AI/ionis-validate/issues>
- IONIS Project: <https://ionis-ai.com/>
