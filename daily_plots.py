from __future__ import annotations

"""Generate daily near-real-time plots for the CRESST cruise.

Typical use:
    python daily_plots.py --date yesterday

Outputs:
- ./output/figs/YYYYMMDD/*.png
- ./output/data/YYYYMMDD/... (cached NetCDF)
- ./output/figs/YYYYMMDD.zip (optional)
"""

import argparse
import os
import datetime as dt
import logging
import shutil
from pathlib import Path
from typing import Dict

import collect as cl
import viz

LOG = logging.getLogger(__name__)


DEFAULT_CMAP_DATASETS: Dict[str, Dict[str, list]] = {
    "tblSST_AVHRR_OI_NRT": {"variables": ["sst"], "units": ["°C"]},
    "tblModis_CHL_NRT": {"variables": ["chlor_a"], "units": ["mg/m³"]},
    "tblModis_PAR_NRT": {"variables": ["PAR"], "units": ["einstein/m²/day"]},
    "tblModis_AOD_REP": {"variables": ["AOD"], "units": [""]},
}


def _parse_date(s: str) -> dt.date:
    s = s.strip().lower()
    if s in {"y", "yesterday"}:
        return dt.date.today() - dt.timedelta(days=1)
    if s in {"t", "today"}:
        return dt.date.today()
    return dt.date.fromisoformat(s)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--date",
        default="yesterday",
        help="Date to plot (YYYY-MM-DD) or 'yesterday' (default).",
    )
    p.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LAT1", "LON1", "LAT2", "LON2"),
        default=[2.0, 120.0, 22.0, 150.0],
        help="Lat/Lon bounding box: lat1 lon1 lat2 lon2.",
    )
    p.add_argument(
        "--stations",
        default="stations.csv",
        help="Path to the CSV with columns lat, lon to overlay sampling stations.",
    )
    p.add_argument(
        "--out",
        default="./output",
        help="Output base directory (default: ./output).",
    )
    p.add_argument(
        "--zip",
        action="store_true",
        default=True,
        help="Create a zip archive of the figure directory.",
    )
    p.add_argument(
        "--copernicus-dataset-version-altimetry",
        default=None,
        help="Optional explicit dataset_version for Copernicus altimetry subset.",
    )
    p.add_argument(
        "--copernicus-dataset-version-wind",
        default=None,
        help="Optional explicit dataset_version for Copernicus wind subset.",
    )
    p.add_argument(
        "--copernicus-dataset-version-biogeochem",
        default=None,
        help="Optional explicit dataset_version for Copernicus biogeochem subset.",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Force Copernicus re-download even if cached NetCDF exists.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return p


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    day = _parse_date(args.date)
    bbox = list(map(float, args.bbox))

    out_base = Path(args.out)
    data_dir = out_base / "data" / day.strftime("%Y%m%d")
    fig_dir = out_base / "figs" / day.strftime("%Y%m%d")
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    stations = None
    if os.path.isfile(args.stations):
        stations = viz.load_stations(args.stations)

    # ---------------- CMAP ----------------
    api = cl.cmap_api()
    for table, props in DEFAULT_CMAP_DATASETS.items():
        for i, var in enumerate(props["variables"]):
            unit = props["units"][i] if i < len(props["units"]) else ""
            df = cl.get_field(table, var, day, bbox, api=api)
            if df.empty:
                LOG.warning("No CMAP data for %s.%s on %s", table, var, day)
                continue
            cl._save_cmap_csv(df, data_dir=data_dir, table=table, variable=var, day=day)
            viz.plot_static_map(df, var, unit, day, fig_dir, stations)

    # ---------------- Copernicus ----------------
    bio = cl.get_biogeochem(
        day,
        bbox,
        data_dir,
        dataset_version=args.copernicus_dataset_version_biogeochem,
        force=args.force_download,
    )
    if not bio.empty:
        viz.plot_static_map(bio, "fe", "mmol/m³", day, fig_dir, stations)
        viz.plot_static_map(bio, "no3", "mmol/m³", day, fig_dir, stations)
        viz.plot_static_map(bio, "po4", "mmol/m³", day, fig_dir, stations)
        viz.plot_static_map(bio, "si", "mmol/m³", day, fig_dir, stations)

    chl = cl.get_plankton(
        day,
        bbox,
        data_dir,
        dataset_version=args.copernicus_dataset_version_altimetry,
        force=args.force_download,
    )
    if not chl.empty:
        viz.plot_static_map(chl, "CHL", "mg/m³", day, fig_dir, stations)

    alt = cl.get_altimetry(
        day,
        bbox,
        data_dir,
        dataset_version=args.copernicus_dataset_version_altimetry,
        force=args.force_download,
    )
    if not alt.empty:
        viz.plot_static_map(alt, "sla", "m", day, fig_dir, stations)
        viz.plot_static_map(alt, "current", "m/s", day, fig_dir, stations)
        viz.plot_vector_field(
            alt,
            u_name="ugos",
            v_name="vgos",
            speed_name="current",
            unit="m/s",
            time_like=day,
            save_path=fig_dir / "quiver" / f"quiver_current_{day}.png",
            stations=stations,
        )

    wind = cl.get_wind(
        day,
        bbox,
        data_dir,
        force=args.force_download,
    )
    if not wind.empty:
        viz.plot_static_map(wind, "wind_speed", "m/s", day, fig_dir, stations)
        viz.plot_vector_field(
            wind,
            u_name="eastward_wind",
            v_name="northward_wind",
            speed_name="wind_speed",
            unit="m/s",
            time_like=day,
            save_path=fig_dir / "quiver" / f"quiver_wind_speed_{day}.png",
            stations=stations,
        )

    if args.zip:
        archive_base = fig_dir.parent / fig_dir.name
        shutil.make_archive(str(archive_base), "zip", str(fig_dir))
        LOG.info("Wrote %s.zip", archive_base)

    LOG.info("CRESST plots saved in %s", fig_dir)
    return 0


if __name__ == "__main__":
    parser = _build_parser()
    raise SystemExit(main(parser.parse_args()))
