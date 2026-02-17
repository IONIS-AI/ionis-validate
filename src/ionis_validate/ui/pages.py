"""
pages.py — Tab content builders for the IONIS V20 browser UI

Each function receives (model, config, checkpoint, device) and builds
its tab content using NiceGUI components. All inference reuses the
existing runner functions — no model logic here.
"""

import json
import logging
import os
import platform
import sys

import numpy as np

log = logging.getLogger("ionis_validate.ui")
import torch
from nicegui import ui

from ionis_validate.model import (
    IonisGate, build_features, grid4_to_latlon, haversine_km, BAND_FREQ_HZ,
)
from ionis_validate.tests.run_predict import (
    sigma_to_approx_db, mode_verdicts, MODE_THRESHOLDS_DB,
)
from ionis_validate.tests.run_adif import (
    parse_adif, extract_observations, run_predictions,
    _MODE_THRESHOLDS, _BAND_MAP, _BAND_MHZ,
)
from ionis_validate.tests.run_report import (
    collect_system_info, generate_report, run_test_suite,
)


# ── Predict Tab ──────────────────────────────────────────────────────────────

def build_predict_tab(model, config, checkpoint, device):
    """Single path prediction form and result display."""
    band_options = list(BAND_FREQ_HZ.keys())

    with ui.card().classes("w-full max-w-2xl mx-auto"):
        ui.label("Single Path Prediction").classes("text-h6 q-mb-md")

        with ui.row().classes("w-full gap-4"):
            tx_grid = ui.input("TX Grid", value="DN46", validation={
                "4-char grid required": lambda v: len(v.strip()) >= 4
            }).classes("w-40")
            rx_grid = ui.input("RX Grid", value="IO91", validation={
                "4-char grid required": lambda v: len(v.strip()) >= 4
            }).classes("w-40")

        with ui.row().classes("w-full gap-4"):
            band = ui.select(options=band_options, label="Band", value="20m").classes("w-32")
            sfi = ui.number("SFI", value=150, min=65, max=350, step=1).classes("w-28")
            kp = ui.number("Kp", value=2.0, min=0, max=9, step=0.5,
                           format="%.1f").classes("w-28")

        with ui.row().classes("w-full gap-4"):
            hour = ui.number("Hour UTC", value=14, min=0, max=23, step=1).classes("w-28")
            month = ui.number("Month", value=6, min=1, max=12, step=1).classes("w-28")

        result_area = ui.column().classes("w-full q-mt-md")

        def run_predict():
            result_area.clear()
            try:
                tx_lat, tx_lon = grid4_to_latlon(tx_grid.value)
                rx_lat, rx_lon = grid4_to_latlon(rx_grid.value)
                freq_hz = BAND_FREQ_HZ[band.value]
                distance_km = haversine_km(tx_lat, tx_lon, rx_lat, rx_lon)

                features = build_features(
                    tx_lat, tx_lon, rx_lat, rx_lon,
                    freq_hz, sfi.value, kp.value, int(hour.value), int(month.value),
                )
                x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    snr_sigma = model(x).item()

                snr_db = sigma_to_approx_db(snr_sigma)
                verdicts = mode_verdicts(snr_db)

                with result_area:
                    with ui.card().classes("w-full bg-blue-50"):
                        ui.label("Prediction Result").classes("text-subtitle1 font-bold")
                        ui.separator()
                        with ui.grid(columns=2).classes("gap-2"):
                            ui.label("TX Grid:")
                            ui.label(f"{tx_grid.value.upper()} ({tx_lat:.1f}N, {tx_lon:.1f}E)")
                            ui.label("RX Grid:")
                            ui.label(f"{rx_grid.value.upper()} ({rx_lat:.1f}N, {rx_lon:.1f}E)")
                            ui.label("Distance:")
                            ui.label(f"{distance_km:,.0f} km")
                            ui.label("Band:")
                            ui.label(f"{band.value} ({freq_hz/1e6:.3f} MHz)")
                            ui.label("Conditions:")
                            ui.label(f"SFI {sfi.value:.0f}, Kp {kp.value:.1f}")
                            ui.label("Time:")
                            ui.label(f"{int(hour.value):02d}:00 UTC, month {int(month.value)}")

                        ui.separator()
                        ui.label(f"Predicted SNR: {snr_sigma:+.3f} sigma ({snr_db:+.1f} dB)").classes(
                            "text-subtitle1 font-bold"
                        )

                        ui.separator()
                        ui.label("Mode Verdicts").classes("text-subtitle2")
                        columns = [
                            {"name": "mode", "label": "Mode", "field": "mode", "align": "left"},
                            {"name": "verdict", "label": "Verdict", "field": "verdict", "align": "center"},
                            {"name": "threshold", "label": "Threshold", "field": "threshold", "align": "right"},
                        ]
                        rows = []
                        for m, v in verdicts.items():
                            rows.append({
                                "mode": m,
                                "verdict": v,
                                "threshold": f"{MODE_THRESHOLDS_DB[m]:+.0f} dB",
                            })
                        ui.table(columns=columns, rows=rows).classes("w-full")

            except Exception as e:
                with result_area:
                    ui.label(f"Error: {e}").classes("text-negative")

        def reset_predict():
            tx_grid.value = "DN46"
            rx_grid.value = "IO91"
            band.value = "20m"
            sfi.value = 150
            kp.value = 2.0
            hour.value = 14
            month.value = 6
            result_area.clear()

        with ui.row().classes("q-mt-sm gap-2"):
            ui.button("Predict", on_click=run_predict, color="primary")
            ui.button("Clear", on_click=reset_predict).props("outline")


# ── Custom Tab ───────────────────────────────────────────────────────────────

def build_custom_tab(model, config, checkpoint, device):
    """Batch custom path tests from JSON file or pasted text."""

    sample_json = json.dumps({
        "description": "Example paths",
        "conditions": {"sfi": 150, "kp": 2},
        "paths": [
            {"tx_grid": "DN46", "rx_grid": "IO91", "band": "20m",
             "hour": 14, "month": 6, "label": "KI7MT to G",
             "expect_open": True, "mode": "WSPR"},
        ],
    }, indent=2)

    with ui.card().classes("w-full max-w-4xl mx-auto"):
        ui.label("Batch Custom Paths").classes("text-h6 q-mb-md")
        ui.label("Paste JSON or upload a .json file").classes("text-body2 text-grey")

        json_input = ui.textarea("JSON", value=sample_json).classes("w-full").props("rows=12")

        async def handle_upload(e):
            data = await e.file.read()
            json_input.value = data.decode("utf-8")

        ui.upload(label="Upload .json", auto_upload=True, on_upload=handle_upload).props(
            'accept=".json" flat'
        ).classes("q-mt-sm")

        result_area = ui.column().classes("w-full q-mt-md")

        def run_custom():
            result_area.clear()
            try:
                spec = json.loads(json_input.value)
            except json.JSONDecodeError as e:
                with result_area:
                    ui.label(f"JSON parse error: {e}").classes("text-negative")
                return

            description = spec.get("description", "Custom paths")
            defaults = spec.get("conditions", {})
            paths = spec.get("paths", [])

            if not paths:
                with result_area:
                    ui.label("No paths defined in JSON").classes("text-negative")
                return

            columns = [
                {"name": "n", "label": "#", "field": "n", "align": "center"},
                {"name": "path", "label": "Path", "field": "path", "align": "left"},
                {"name": "band", "label": "Band", "field": "band", "align": "center"},
                {"name": "db", "label": "dB", "field": "db", "align": "right"},
                {"name": "km", "label": "km", "field": "km", "align": "right"},
                {"name": "sfi", "label": "SFI", "field": "sfi", "align": "right"},
                {"name": "kp", "label": "Kp", "field": "kp", "align": "right"},
                {"name": "hour", "label": "Hour", "field": "hour", "align": "right"},
                {"name": "month", "label": "Mon", "field": "month", "align": "right"},
                {"name": "wspr", "label": "WSPR", "field": "wspr", "align": "center"},
                {"name": "ft8", "label": "FT8", "field": "ft8", "align": "center"},
                {"name": "cw", "label": "CW", "field": "cw", "align": "center"},
                {"name": "ssb", "label": "SSB", "field": "ssb", "align": "center"},
                {"name": "result", "label": "Result", "field": "result", "align": "center"},
            ]

            rows = []
            pass_count = 0
            fail_count = 0
            display_modes = ["WSPR", "FT8", "CW", "SSB"]

            for i, p in enumerate(paths):
                tx_g = p["tx_grid"].upper()
                rx_g = p["rx_grid"].upper()
                b = p["band"]
                path_sfi = p.get("sfi", defaults.get("sfi", 150))
                path_kp = p.get("kp", defaults.get("kp", 2))
                path_hour = p.get("hour", defaults.get("hour", 12))
                path_month = p.get("month", defaults.get("month", 6))
                expect_open = p.get("expect_open", None)
                test_mode = p.get("mode", "WSPR").upper()

                if b not in BAND_FREQ_HZ:
                    rows.append({
                        "n": str(i + 1), "path": f"{tx_g}>{rx_g}", "band": b,
                        "db": "--", "km": "--", "sfi": str(int(path_sfi)),
                        "kp": f"{path_kp:.1f}", "hour": str(path_hour),
                        "month": str(path_month),
                        "wspr": "--", "ft8": "--", "cw": "--", "ssb": "--",
                        "result": "SKIP",
                    })
                    continue

                tx_lat, tx_lon = grid4_to_latlon(tx_g)
                rx_lat, rx_lon = grid4_to_latlon(rx_g)
                freq_hz = BAND_FREQ_HZ[b]
                distance_km = haversine_km(tx_lat, tx_lon, rx_lat, rx_lon)

                features = build_features(
                    tx_lat, tx_lon, rx_lat, rx_lon,
                    freq_hz, path_sfi, path_kp, path_hour, path_month,
                )
                x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    snr_sigma = model(x).item()

                snr_db = sigma_to_approx_db(snr_sigma)

                mode_cols = {}
                for m in display_modes:
                    mode_cols[m] = "OPEN" if snr_db >= MODE_THRESHOLDS_DB[m] else "--"

                if expect_open is not None and test_mode in MODE_THRESHOLDS_DB:
                    threshold = MODE_THRESHOLDS_DB[test_mode]
                    is_open = snr_db >= threshold
                    if expect_open == is_open:
                        status = "PASS"
                        pass_count += 1
                    else:
                        status = "FAIL"
                        fail_count += 1
                else:
                    is_open = snr_db >= MODE_THRESHOLDS_DB["WSPR"]
                    status = "OPEN" if is_open else "closed"

                rows.append({
                    "n": str(i + 1), "path": f"{tx_g}>{rx_g}", "band": b,
                    "db": f"{snr_db:+.1f}", "km": f"{distance_km:,.0f}",
                    "sfi": str(int(path_sfi)), "kp": f"{path_kp:.1f}",
                    "hour": str(path_hour), "month": str(path_month),
                    "wspr": mode_cols["WSPR"], "ft8": mode_cols["FT8"],
                    "cw": mode_cols["CW"], "ssb": mode_cols["SSB"],
                    "result": status,
                })

            with result_area:
                ui.label(f"{description}").classes("text-subtitle1 font-bold")
                ui.table(columns=columns, rows=rows).classes("w-full")

                has_expectations = (pass_count + fail_count) > 0
                if has_expectations:
                    total_tested = pass_count + fail_count
                    color = "text-positive" if fail_count == 0 else "text-negative"
                    ui.label(
                        f"Expectations: {pass_count}/{total_tested} passed"
                    ).classes(f"text-subtitle2 {color} q-mt-sm")

        def reset_custom():
            json_input.value = sample_json
            result_area.clear()

        with ui.row().classes("q-mt-sm gap-2"):
            ui.button("Run", on_click=run_custom, color="primary")
            ui.button("Clear", on_click=reset_custom).props("outline")


# ── ADIF Tab ─────────────────────────────────────────────────────────────────

def build_adif_tab(model, config, checkpoint, device):
    """ADIF log validation with file upload."""
    band_names = {102: "160m", 103: "80m", 104: "60m", 105: "40m", 106: "30m",
                  107: "20m", 108: "17m", 109: "15m", 110: "12m", 111: "10m"}
    mode_labels = {"DG": "Digital", "CW": "CW", "RY": "RTTY", "PH": "SSB"}

    with ui.card().classes("w-full max-w-3xl mx-auto"):
        ui.label("ADIF Log Validation").classes("text-h6 q-mb-md")
        ui.label(
            "Upload your ADIF (.adi/.adif) log. Callsigns are stripped at parse time "
            "and never stored."
        ).classes("text-body2 text-grey q-mb-sm")

        upload_state = {"path": None, "name": None}

        async def handle_upload(e):
            try:
                log.info("[ADIF] handle_upload fired, file=%s", e.file.name)
                # Clean up previous temp file
                if upload_state["path"] and os.path.exists(upload_state["path"]):
                    os.unlink(upload_state["path"])
                data = await e.file.read()
                import tempfile
                tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".adi", delete=False)
                tmp.write(data)
                tmp.close()
                upload_state["path"] = tmp.name
                upload_state["name"] = e.file.name
                log.info("[ADIF] saved %d bytes to %s", len(data), tmp.name)
                status_label.set_text(f"Loaded: {e.file.name} ({len(data):,} bytes)")
            except Exception as ex:
                log.error("[ADIF] upload error: %s", ex)
                status_label.set_text(f"Upload error: {ex}")

        def handle_rejected():
            log.warning("[ADIF] file rejected by browser (size or type)")
            status_label.set_text("File rejected — must be .adi/.adif, max 100 MB")

        ui.upload(
            label="Upload .adi / .adif", auto_upload=True,
            on_upload=handle_upload, on_rejected=handle_rejected,
            max_file_size=100_000_000,
        ).props('accept=".adi,.adif"').classes("w-full")
        status_label = ui.label("").classes("text-body2")

        result_area = ui.column().classes("w-full q-mt-md")

        async def run_adif():
            result_area.clear()
            log.info("[ADIF] run_adif called, upload_state=%r", upload_state)

            if upload_state["path"] is None:
                with result_area:
                    ui.label("No file uploaded").classes("text-negative")
                return

            tmp_path = upload_state["path"]
            log.info("[ADIF] parsing %s", tmp_path)

            try:
                records = parse_adif(tmp_path)
                observations, skipped = extract_observations(records)

                if not observations:
                    with result_area:
                        ui.label(
                            f"Parsed {len(records):,} records but no valid HF observations found."
                        ).classes("text-negative")
                    return

                results, solar_hits = run_predictions(
                    observations, model, config, device,
                )

                total = len(results)
                open_count = sum(1 for r in results if r["band_open"])
                recall = 100.0 * open_count / total

                with result_area:
                    # Summary card
                    with ui.card().classes("w-full bg-blue-50"):
                        ui.label("Validation Results").classes("text-subtitle1 font-bold")
                        ui.separator()
                        solar_pct = 100.0 * solar_hits / total if total else 0
                        with ui.grid(columns=2).classes("gap-2"):
                            ui.label("Log file:")
                            ui.label(upload_state["name"])
                            ui.label("Solar conditions:")
                            ui.label(f"{solar_hits:,}/{total:,} ({solar_pct:.0f}%) per-QSO daily lookup")
                            ui.label("Observations:")
                            ui.label(f"{total:,}")
                            ui.label("Band open:")
                            ui.label(f"{open_count:,}")

                        ui.separator()
                        recall_color = "text-positive" if recall >= 50 else "text-warning"
                        ui.label(f"Recall: {recall:.2f}%").classes(
                            f"text-h6 font-bold {recall_color}"
                        )

                    # Skipped summary
                    skip_labels = {
                        "no_grid": "Missing grid square",
                        "no_band": "No recognized band",
                        "bad_grid": "Invalid grid format",
                        "non_hf": "Non-HF band",
                        "no_solar": "Before 2000 (no solar data)",
                    }
                    total_skipped = sum(skipped.values())
                    if total_skipped:
                        with ui.card().classes("w-full q-mt-sm"):
                            ui.label(f"Records skipped: {total_skipped:,}").classes("text-subtitle2")
                            for reason, count in skipped.items():
                                if count > 0:
                                    label = skip_labels.get(reason, reason)
                                    ui.label(f"  {label}: {count:,}").classes("text-body2")

                    # Recall by mode
                    modes_seen = sorted(set(r["ionis_mode"] for r in results))
                    if modes_seen:
                        mode_columns = [
                            {"name": "mode", "label": "Mode", "field": "mode", "align": "left"},
                            {"name": "recall", "label": "Recall", "field": "recall", "align": "right"},
                            {"name": "count", "label": "QSOs", "field": "count", "align": "right"},
                        ]
                        mode_rows = []
                        for m in modes_seen:
                            m_results = [r for r in results if r["ionis_mode"] == m]
                            m_open = sum(1 for r in m_results if r["band_open"])
                            m_recall = 100.0 * m_open / len(m_results) if m_results else 0
                            label = mode_labels.get(m, m)
                            mode_rows.append({
                                "mode": label, "recall": f"{m_recall:.2f}%",
                                "count": f"{len(m_results):,}",
                            })
                        ui.label("Recall by Mode").classes("text-subtitle2 q-mt-md")
                        ui.table(columns=mode_columns, rows=mode_rows).classes("w-full")

                    # Recall by band
                    band_columns = [
                        {"name": "band", "label": "Band", "field": "band", "align": "left"},
                        {"name": "recall", "label": "Recall", "field": "recall", "align": "right"},
                        {"name": "count", "label": "QSOs", "field": "count", "align": "right"},
                    ]
                    band_rows = []
                    for bid in sorted(band_names.keys()):
                        b_results = [r for r in results if r["band_id"] == bid]
                        if not b_results:
                            continue
                        b_open = sum(1 for r in b_results if r["band_open"])
                        b_recall = 100.0 * b_open / len(b_results)
                        band_rows.append({
                            "band": band_names[bid], "recall": f"{b_recall:.2f}%",
                            "count": f"{len(b_results):,}",
                        })
                    if band_rows:
                        ui.label("Recall by Band").classes("text-subtitle2 q-mt-md")
                        ui.table(columns=band_columns, rows=band_rows).classes("w-full")

                    # SNR Pearson if available
                    snr_results = [r for r in results if "snr_db" in r]
                    if len(snr_results) >= 10:
                        actual = np.array([r["snr_db"] for r in snr_results])
                        predicted = np.array([r["predicted_db"] for r in snr_results])
                        correlation = np.corrcoef(actual, predicted)[0, 1]
                        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                        with ui.card().classes("w-full q-mt-sm bg-amber-50"):
                            ui.label(
                                f"SNR Comparison ({len(snr_results):,} QSOs with signal reports)"
                            ).classes("text-subtitle2")
                            ui.label(f"Pearson: {correlation:+.4f}").classes("text-body1")
                            ui.label(f"RMSE: {rmse:.1f} dB").classes("text-body1")

            except Exception as e:
                with result_area:
                    ui.label(f"Error: {e}").classes("text-negative")

        def reset_adif():
            if upload_state["path"] and os.path.exists(upload_state["path"]):
                os.unlink(upload_state["path"])
            upload_state["path"] = None
            upload_state["name"] = None
            status_label.set_text("")
            result_area.clear()

        with ui.row().classes("q-mt-sm gap-2"):
            ui.button("Validate", on_click=run_adif, color="primary")
            ui.button("Clear", on_click=reset_adif).props("outline")


# ── Info Tab ─────────────────────────────────────────────────────────────────

def build_info_tab(model, config, checkpoint, device):
    """Model version, checkpoint metrics, and system info."""
    param_count = sum(p.numel() for p in model.parameters())

    with ui.card().classes("w-full max-w-2xl mx-auto"):
        ui.label("Model Information").classes("text-h6 q-mb-md")

        # Model section
        ui.label("MODEL").classes("text-subtitle2 font-bold q-mt-sm")
        ui.separator()
        info_columns = [
            {"name": "key", "label": "Property", "field": "key", "align": "left"},
            {"name": "value", "label": "Value", "field": "value", "align": "left"},
        ]
        model_rows = [
            {"key": "Version", "value": f"{config['version']} ({config['phase']})"},
            {"key": "Architecture", "value": config["model"]["architecture"]},
            {"key": "DNN dim", "value": str(config["model"]["dnn_dim"])},
            {"key": "Hidden dim", "value": str(config["model"]["hidden_dim"])},
            {"key": "Input dim", "value": str(config["model"]["input_dim"])},
            {"key": "Sidecar hidden", "value": str(config["model"]["sidecar_hidden"])},
            {"key": "Parameters", "value": f"{param_count:,}"},
        ]
        ui.table(columns=info_columns, rows=model_rows).classes("w-full")

        # Checkpoint section
        ui.label("CHECKPOINT").classes("text-subtitle2 font-bold q-mt-lg")
        ui.separator()
        ckpt_rows = []
        epoch = checkpoint.get("epoch")
        if epoch is not None:
            ckpt_rows.append({"key": "Epoch", "value": str(epoch)})
        val_pearson = checkpoint.get("val_pearson")
        if val_pearson is not None:
            ckpt_rows.append({"key": "Pearson", "value": f"{val_pearson:+.4f}"})
        val_rmse = checkpoint.get("val_rmse")
        if val_rmse is not None:
            ckpt_rows.append({"key": "RMSE", "value": f"{val_rmse:.4f} sigma"})
        date_range = checkpoint.get("date_range")
        if date_range:
            ckpt_rows.append({"key": "Date range", "value": str(date_range)})
        sample_size = checkpoint.get("sample_size")
        if sample_size:
            ckpt_rows.append({"key": "Sample size", "value": f"{sample_size:,}"})
        ui.table(columns=info_columns, rows=ckpt_rows).classes("w-full")

        # Features
        features = config.get("features", [])
        if features:
            ui.label(f"FEATURES ({len(features)})").classes("text-subtitle2 font-bold q-mt-lg")
            ui.separator()
            feat_columns = [
                {"name": "idx", "label": "Index", "field": "idx", "align": "center"},
                {"name": "name", "label": "Feature", "field": "name", "align": "left"},
            ]
            feat_rows = [{"idx": str(i), "name": f} for i, f in enumerate(features)]
            ui.table(columns=feat_columns, rows=feat_rows).classes("w-full")

        # System section
        ui.label("SYSTEM").classes("text-subtitle2 font-bold q-mt-lg")
        ui.separator()
        sys_rows = [
            {"key": "Python", "value": sys.version.split()[0]},
            {"key": "PyTorch", "value": torch.__version__},
            {"key": "Device", "value": str(device)},
            {"key": "Platform", "value": f"{platform.system()} {platform.machine()}"},
            {"key": "Hostname", "value": platform.node()},
        ]
        if device.type == "cuda":
            sys_rows.append({"key": "CUDA version", "value": torch.version.cuda})
            sys_rows.append({"key": "GPU", "value": torch.cuda.get_device_name(0)})
        elif device.type == "mps":
            sys_rows.append({"key": "MPS", "value": "available"})
        ui.table(columns=info_columns, rows=sys_rows).classes("w-full")


# ── Report Tab ───────────────────────────────────────────────────────────────

GITHUB_NEW_ISSUE = "https://github.com/IONIS-AI/ionis-validate/issues/new/choose"


def build_report_tab(model, config, checkpoint, device):
    """Generate a beta test report for GitHub Issues."""
    with ui.card().classes("w-full max-w-3xl mx-auto"):
        ui.label("Beta Test Report").classes("text-h6 q-mb-md")
        ui.label(
            "Generate a structured report you can paste into a GitHub Issue. "
            "Collects system info and optionally runs the 62-test suite."
        ).classes("text-body2 text-grey q-mb-sm")

        run_tests_toggle = ui.switch("Include test suite results", value=True)

        result_area = ui.column().classes("w-full q-mt-md")

        def generate():
            result_area.clear()

            with result_area:
                ui.label("Collecting system info...").classes("text-body2")

            info = collect_system_info()
            if "error" in info:
                result_area.clear()
                with result_area:
                    ui.label(f"Error: {info['error']}").classes("text-negative")
                return

            test_passed = None
            test_output = None

            if run_tests_toggle.value:
                result_area.clear()
                with result_area:
                    ui.label("Running 62-test suite... this may take a moment.").classes(
                        "text-body2"
                    )
                test_passed, test_output = run_test_suite()

            report = generate_report(
                info,
                test_passed=test_passed,
                test_output=test_output,
            )

            result_area.clear()
            with result_area:
                status_text = ""
                if test_passed is not None:
                    status_text = " — Tests: PASS" if test_passed else " — Tests: FAIL"
                    color = "text-positive" if test_passed else "text-negative"
                    ui.label(f"Report generated{status_text}").classes(
                        f"text-subtitle1 font-bold {color}"
                    )
                else:
                    ui.label("Report generated (tests skipped)").classes(
                        "text-subtitle1 font-bold"
                    )

                report_area = ui.textarea(
                    "Report Markdown", value=report,
                ).classes("w-full font-mono").props("rows=20 readonly")

                with ui.row().classes("gap-2 q-mt-sm"):
                    ui.button(
                        "Copy to Clipboard",
                        on_click=lambda: ui.run_javascript(
                            f'navigator.clipboard.writeText({json.dumps(report)})'
                        ),
                        color="primary",
                    )
                    ui.link(
                        "Open GitHub Issues",
                        target=GITHUB_NEW_ISSUE,
                        new_tab=True,
                    ).classes("q-btn q-btn--outline text-primary q-mt-xs")

        def reset_report():
            run_tests_toggle.value = True
            result_area.clear()

        with ui.row().classes("q-mt-sm gap-2"):
            ui.button("Generate Report", on_click=generate, color="primary")
            ui.button("Clear", on_click=reset_report).props("outline")
