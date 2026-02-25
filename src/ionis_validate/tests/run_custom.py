#!/usr/bin/env python3
"""
run_custom.py — IONIS V22-gamma Batch Custom Path Tests

Run predictions for a set of user-defined paths from a JSON file.
Includes PhysicsOverrideLayer and day_of_year support.

Usage:
  python run_custom.py my_paths.json
  ionis-validate custom my_paths.json

JSON format:
  {
    "description": "My 20m paths from KI7MT",
    "conditions": { "sfi": 140, "kp": 1.5 },
    "paths": [
      {
        "tx_grid": "DN26", "rx_grid": "IO91", "band": "20m",
        "hour": 14, "month": 6, "day_of_year": 172,
        "label": "KI7MT to G"
      },
      {
        "tx_grid": "DN26", "rx_grid": "PM95", "band": "20m",
        "hour": 6, "month": 12, "day_of_year": 355,
        "label": "KI7MT to JA",
        "expect_open": true, "mode": "CW"
      }
    ]
  }

Per-path overrides for sfi/kp/hour/month/day_of_year take precedence over
top-level "conditions" defaults. Optional "expect_open" (bool) triggers
pass/fail against the specified "mode" threshold (default: WSPR).
"""

import argparse
import json
import sys

import torch

from ionis_validate.model import (
    get_device, load_model,
    grid4_to_latlon, build_features, haversine_km,
    solar_elevation_deg, BAND_FREQ_HZ,
)
from ionis_validate.physics_override import apply_override_to_prediction

# ── Constants ────────────────────────────────────────────────────────────────

WSPR_MEAN_DB = -17.53
WSPR_STD_DB = 6.7

# Mode decode thresholds (dB)
MODE_THRESHOLDS_DB = {
    "WSPR": -28.0,
    "FT8":  -21.0,
    "CW":   -15.0,
    "RTTY":  -5.0,
    "SSB":    3.0,
}

# Modes shown as inline columns
DISPLAY_MODES = ["WSPR", "FT8", "CW", "SSB"]

# Table row format
ROW_FMT = (
    "    {n:>3s}  {path:<13s}  {band:<4s}"
    "  {db:>6s}  {km:>7s}"
    "  {sfi:>3s}  {kp:>3s}  {hour:>4s}  {mon:>3s}  {doy:>3s}"
    "  {wspr:>4s}  {ft8:>4s}  {cw:>4s}  {ssb:>4s}"
    "  {ovr:>3s}  {result:>6s}"
)


def sigma_to_approx_db(sigma):
    return sigma * WSPR_STD_DB + WSPR_MEAN_DB


def format_kp(kp):
    """Format Kp: integer if whole number, one decimal otherwise."""
    if kp == int(kp):
        return str(int(kp))
    return f"{kp:.1f}"


def main():
    parser = argparse.ArgumentParser(
        description="IONIS V22-gamma — Batch custom path tests with PhysicsOverrideLayer",
    )
    parser.add_argument("json_file", nargs="?", default=None,
                        help="Path to JSON file defining custom paths")
    parser.add_argument("--example", action="store_true",
                        help="Run the bundled example (2 easy, 2 medium, 2 hard)")
    parser.add_argument("--config", default=None, help="Path to config JSON")
    parser.add_argument("--checkpoint", default=None, help="Path to .safetensors file")

    args = parser.parse_args()

    if args.example:
        from ionis_validate import _data_path
        json_path = _data_path("example_custom_paths.json")
    elif args.json_file:
        json_path = args.json_file
    else:
        print("  ERROR: Provide a JSON file or use --example\n")
        print("  Usage:")
        print("    ionis-validate custom my_paths.json")
        print("    ionis-validate custom --example")
        return 1

    # Load test spec
    with open(json_path) as f:
        spec = json.load(f)

    description = spec.get("description", "Custom paths")
    defaults = spec.get("conditions", {})
    paths = spec.get("paths", [])

    if not paths:
        print("  ERROR: No paths defined in JSON file", file=sys.stderr)
        return 1

    # Load model
    device = get_device()
    model, config, metadata = load_model(args.config, args.checkpoint, device)

    # Header
    print()
    print("=" * 70)
    print("  IONIS V22-gamma — Custom Path Tests")
    print("=" * 70)
    print(f"\n  {description}")
    print(f"  Device: {device}")
    print()

    # Table header
    hdr = ROW_FMT.format(
        n="#", path="Path", band="Band", db="dB", km="km",
        sfi="SFI", kp="Kp", hour="Hour", mon="Mon", doy="DoY",
        wspr="WSPR", ft8="FT8", cw="CW", ssb="SSB",
        ovr="OVR", result="Result",
    )
    print(hdr)
    sep_width = len(hdr)
    print(f"  {'─' * (sep_width - 2)}")

    # Run predictions
    pass_count = 0
    fail_count = 0
    tested_modes = set()

    for i, p in enumerate(paths):
        tx_grid = p["tx_grid"].upper()
        rx_grid = p["rx_grid"].upper()
        band = p["band"]
        path_str = f"{tx_grid} > {rx_grid}"

        # Per-path overrides or defaults
        sfi = p.get("sfi", defaults.get("sfi", 150))
        kp = p.get("kp", defaults.get("kp", 2))
        hour = p.get("hour", defaults.get("hour", 12))
        month = p.get("month", defaults.get("month", 6))
        day_of_year = p.get("day_of_year", defaults.get("day_of_year", 172))
        expect_open = p.get("expect_open", None)
        test_mode = p.get("mode", "WSPR").upper()

        # Validate mode
        if test_mode not in MODE_THRESHOLDS_DB:
            print(f"  SKIP: Unknown mode '{test_mode}' for path #{i+1}")
            continue

        if band not in BAND_FREQ_HZ:
            print(ROW_FMT.format(
                n=str(i + 1), path=path_str, band=band,
                db="—", km="—",
                sfi=f"{sfi:.0f}", kp=format_kp(kp),
                hour=str(hour), mon=str(month), doy=str(day_of_year),
                wspr="—", ft8="—", cw="—", ssb="—",
                ovr="—", result="SKIP",
            ))
            continue

        tx_lat, tx_lon = grid4_to_latlon(tx_grid)
        rx_lat, rx_lon = grid4_to_latlon(rx_grid)
        freq_hz = BAND_FREQ_HZ[band]
        freq_mhz = freq_hz / 1e6
        distance_km = haversine_km(tx_lat, tx_lon, rx_lat, rx_lon)

        features = build_features(
            tx_lat, tx_lon, rx_lat, rx_lon,
            freq_hz, sfi, kp, hour, month,
            day_of_year=day_of_year,
            include_solar_depression=True,
        )
        x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            raw_sigma = model(x).item()

        # PhysicsOverrideLayer
        tx_solar = solar_elevation_deg(tx_lat, tx_lon, hour, day_of_year)
        rx_solar = solar_elevation_deg(rx_lat, rx_lon, hour, day_of_year)
        snr_sigma, was_overridden = apply_override_to_prediction(
            raw_sigma, freq_mhz, tx_solar, rx_solar)

        snr_db = sigma_to_approx_db(snr_sigma)

        # Mode verdicts for display columns
        mode_cols = {}
        for m in DISPLAY_MODES:
            mode_cols[m] = "OPEN" if snr_db >= MODE_THRESHOLDS_DB[m] else "--"

        ovr_str = "YES" if was_overridden else ""

        # Determine result
        if expect_open is not None:
            threshold = MODE_THRESHOLDS_DB[test_mode]
            is_open = snr_db >= threshold
            if expect_open == is_open:
                status = "PASS"
                pass_count += 1
            else:
                status = "FAIL"
                fail_count += 1
            tested_modes.add(test_mode)
        else:
            # No expectation — show OPEN/closed based on WSPR threshold
            is_open = snr_db >= MODE_THRESHOLDS_DB["WSPR"]
            status = "OPEN" if is_open else "closed"

        # Print row
        print(ROW_FMT.format(
            n=str(i + 1), path=path_str, band=band,
            db=f"{snr_db:+.1f}", km=f"{distance_km:,.0f}",
            sfi=f"{sfi:.0f}", kp=format_kp(kp),
            hour=str(hour), mon=str(month), doy=str(day_of_year),
            wspr=mode_cols["WSPR"], ft8=mode_cols["FT8"],
            cw=mode_cols["CW"], ssb=mode_cols["SSB"],
            ovr=ovr_str, result=status,
        ))

    print(f"  {'─' * (sep_width - 2)}")

    # Summary
    has_expectations = (pass_count + fail_count) > 0
    if has_expectations:
        total_tested = pass_count + fail_count
        if len(tested_modes) == 1:
            mode_note = f" (mode: {next(iter(tested_modes))})"
        else:
            mode_note = ""
        print(f"\n  Expectations: {pass_count}/{total_tested} passed{mode_note}")
        if fail_count > 0:
            print(f"  {fail_count} path(s) did not match expected open/closed state")

    print()
    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
