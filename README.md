# ionis-validate

Validation suite for the IONIS V22-gamma + PhysicsOverrideLayer HF propagation
model. Run 29 physics tests (18 KI7MT operator-grounded + 11 band x time
discrimination), predict SNR for any HF path, or batch-test your own paths —
all from the command line or a browser UI, on any platform.

IONIS (Ionospheric Neural Inference System) predicts HF radio signal
strength from WSPR, RBN, and contest data. The V22-gamma model was trained on
38.7 million propagation observations spanning 2008-2026, with a deterministic
PhysicsOverrideLayer for high-band night closure.

## Install

```
pip install ionis-validate
```

Requires Python 3.10+ and PyTorch 2.0+. Works on Windows, macOS, and Linux.

## Quick Start

```bash
# Show model and system info
ionis-validate info

# Run the full 29-test validation suite
ionis-validate test

# Predict SNR for a single path
ionis-validate predict \
    --tx-grid FN20 --rx-grid IO91 --band 20m \
    --sfi 150 --kp 2 --hour 14 --month 6 --day-of-year 172

# Acid test: 10m EU path at night (override should fire)
ionis-validate predict \
    --tx-grid DN46 --rx-grid JN48 --band 10m \
    --sfi 150 --kp 2 --hour 2 --month 2 --day-of-year 45
```

## Browser UI

A point-and-click dashboard wrapping every command.

```
pip install "ionis-validate[ui]"
ionis-validate ui
```

Opens a browser tab at `http://localhost:8765` with tabs for Predict,
Custom, Report, and Info.

## Batch Predictions

Define a set of paths in a JSON file and run them all at once:

```bash
# Run the bundled example (2 easy, 2 medium, 2 hard)
ionis-validate custom --example

# Run your own paths
ionis-validate custom my_paths.json
```

See the [Custom Path Tests](https://ionis-ai.com/testing/custom-paths/)
documentation for the JSON format. V22 adds `day_of_year` and an OVR
(override) column to the output.

## Beta Testing

Generate a structured report for filing as a GitHub Issue:

```bash
ionis-validate report
```

## Privacy

IONIS processes only grid-pair geometry, band, time, and solar indices.
No callsigns, names, or personal data are used by the model or stored
by this tool.

Full privacy policy: <https://ionis-ai.com/about/data-privacy/>

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

## Links

- Documentation: <https://ionis-ai.com/testing/>
- Source: <https://github.com/IONIS-AI/ionis-validate>
- Issues: <https://github.com/IONIS-AI/ionis-validate/issues>
- IONIS Project: <https://ionis-ai.com/>
