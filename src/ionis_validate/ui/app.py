"""
app.py — IONIS V20 Browser-based Validation Dashboard

Launches a NiceGUI app on localhost:8765 wrapping the existing
ionis-validate runner functions. No model logic changes.

Usage:
  ionis-validate ui
"""

import logging
import sys


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if sys.version_info < (3, 10):
        print()
        print(f"  ERROR: The browser UI requires Python >= 3.10 (you have {sys.version.split()[0]})")
        print()
        return 1

    try:
        from nicegui import app, ui
    except ImportError:
        print()
        print("  ERROR: NiceGUI is not installed.")
        print()
        print("  The browser UI requires optional dependencies. Install with:")
        print()
        print('    pip install "ionis-validate[ui]"')
        print()
        return 1

    from ionis_validate.model import load_model, get_device, IonisGate
    from ionis_validate.ui.pages import (
        build_predict_tab,
        build_custom_tab,
        build_info_tab,
        build_report_tab,
    )

    # Load model once at startup
    device = get_device()
    model, config, checkpoint = load_model(device=device)
    shared = (model, config, checkpoint, device)

    @ui.page("/")
    def index():
        # Theme
        ui.colors(primary="#3F51B5", secondary="#FFC107", accent="#FFC107")
        dark = ui.dark_mode(value=False)

        # Header
        with ui.header().classes("items-center justify-between"):
            ui.label("IONIS V20").classes("text-h6 font-bold")
            with ui.row().classes("items-center"):
                ui.label("Dark mode").classes("text-body2")
                ui.switch(on_change=lambda e: dark.set_value(e.value))

        # Tabs
        with ui.tabs().classes("w-full") as tabs:
            predict_tab = ui.tab("Predict")
            custom_tab = ui.tab("Custom")
            report_tab = ui.tab("Report")
            info_tab = ui.tab("Info")

        with ui.tab_panels(tabs, value=predict_tab).classes("w-full"):
            with ui.tab_panel(predict_tab):
                build_predict_tab(*shared)
            with ui.tab_panel(custom_tab):
                build_custom_tab(*shared)
            with ui.tab_panel(report_tab):
                build_report_tab(*shared)
            with ui.tab_panel(info_tab):
                build_info_tab(*shared)

    print()
    print("  IONIS V20 — Browser UI starting on http://0.0.0.0:8765")
    print("  Open http://<this-machine>:8765 in your browser")
    print()

    ui.run(host="0.0.0.0", port=8765, title="IONIS V20", reload=False)
    return 0
