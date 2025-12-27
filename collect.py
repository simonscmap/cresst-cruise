from __future__ import annotations

"""Data acquisition helpers for daily CRESST cruise visualizations.

This module pulls near-real-time gridded fields from:

- Simons CMAP via ``pycmap``
- Copernicus Marine via ``copernicusmarine``

Credentials
-----------
Prefer environment variables:

- ``CMAP_API_KEY``
- ``COPERNICUSMARINE_USERNAME``
- ``COPERNICUSMARINE_PASSWORD``

If those are not present, we fall back to importing a local ``credentials.py``
module with attributes: ``cmap_key``, ``usr_cmem``, ``psw_cmem``.

Caching
-------
Copernicus NetCDF downloads are cached on disk. If a requested file already
exists, it is reused (no re-download).
"""

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import copernicusmarine
import pandas as pd
import pycmap
import xarray as xr

LOG = logging.getLogger(__name__)

LatLonBox = Tuple[float, float, float, float]  # (lat1, lon1, lat2, lon2)


def _as_date(dt: Union[date, datetime]) -> date:
    return dt.date() if isinstance(dt, datetime) else dt


def _ymd(dt: Union[date, datetime]) -> str:
    d = _as_date(dt)
    return f"{d.year:04d}-{d.month:02d}-{d.day:02d}"


def _ymd_compact(dt: Union[date, datetime]) -> str:
    d = _as_date(dt)
    return f"{d.year:04d}{d.month:02d}{d.day:02d}"


def _load_local_credentials():
    """Import local credentials.py only if it exists."""
    try:
        import credentials as cr  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing credentials. Set CMAP_API_KEY / COPERNICUSMARINE_USERNAME / "
            "COPERNICUSMARINE_PASSWORD, or create a local credentials.py (see README)."
        ) from e
    return cr


def load_cmap_key() -> str:
    key = os.environ.get("CMAP_API_KEY")
    if key:
        return key.strip()
    cr = _load_local_credentials()
    return str(getattr(cr, "cmap_key")).strip()


def load_copernicus_creds() -> Tuple[str, str]:
    usr = os.environ.get("COPERNICUSMARINE_USERNAME")
    psw = os.environ.get("COPERNICUSMARINE_PASSWORD")
    if usr and psw:
        return usr.strip(), psw
    cr = _load_local_credentials()
    return str(getattr(cr, "usr_cmem")).strip(), str(getattr(cr, "psw_cmem"))


def cmap_api(cmap_key: Optional[str] = None) -> pycmap.API:
    return pycmap.API(cmap_key or load_cmap_key())


# ----------------------- CMAP ----------------------- #


def get_field(
    table: str,
    variable: str,
    dt: Union[date, datetime],
    domain: Sequence[float],
    *,
    depth: Tuple[float, float] = (0.0, 0.0),
    api: Optional[pycmap.API] = None,
) -> pd.DataFrame:
    """Fetch a CMAP field for a single day as a long-form dataframe."""
    dts = _ymd(dt)
    lat1, lon1, lat2, lon2 = map(float, domain)
    api = api or cmap_api()

    try:
        df = api.space_time(
            table=table,
            variable=variable,
            dt1=dts,
            dt2=dts,
            lat1=lat1,
            lat2=lat2,
            lon1=lon1,
            lon2=lon2,
            depth1=float(depth[0]),
            depth2=float(depth[1]),
        )
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    except Exception:
        LOG.exception("CMAP request failed for %s.%s on %s", table, variable, dts)
        return pd.DataFrame()


def get_climatology(
    table: str,
    variable: str,
    month: int,
    domain: Sequence[float],
    *,
    api: Optional[pycmap.API] = None,
) -> pd.DataFrame:
    """Fetch a CMAP monthly climatology (avg over all years) for a domain."""
    if not (1 <= int(month) <= 12):
        raise ValueError("month must be in [1, 12]")

    lat1, lon1, lat2, lon2 = map(float, domain)
    api = api or cmap_api()

    q = (
        f"select avg({variable}) as {variable}, lat, lon "
        f"from {table} "
        f"where month={int(month)} "
        f"and lat>={lat1} and lat<={lat2} and lon>={lon1} and lon<={lon2} "
        f"group by lat, lon"
    )

    try:
        df = api.query(q)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    except Exception:
        LOG.exception("CMAP climatology query failed for %s.%s month=%s", table, variable, month)
        return pd.DataFrame()


def last_available_date(
    table: str,
    variable: str,
    *,
    api: Optional[pycmap.API] = None,
) -> Optional[datetime]:
    """Return CMAP's latest available timestamp for (table, variable)."""
    api = api or cmap_api()
    try:
        cov = api.get_var_coverage(table, variable)
        max_dt = pd.to_datetime(cov["Time_Max"][0]).to_pydatetime()
        today = datetime.now().date()
        safe_day = min(max_dt.date(), today)
        return datetime(
            safe_day.year,
            safe_day.month,
            safe_day.day,
            max_dt.hour,
            max_dt.minute,
            max_dt.second,
        )
    except Exception:
        LOG.exception("Failed to query CMAP coverage for %s.%s", table, variable)
        return None


# ----------------------- Copernicus Marine ----------------------- #


def netcdf_to_dataframe(nc_path: Union[str, Path], *, variables: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Convert a NetCDF file to a dataframe (coords + selected variables)."""
    nc_path = Path(nc_path)
    ds = xr.open_dataset(nc_path)
    try:
        if variables:
            ds = ds[list(variables)]
        df = ds.to_dataframe().reset_index()
    finally:
        ds.close()

    rename = {}
    if "latitude" in df.columns:
        rename["latitude"] = "lat"
    if "longitude" in df.columns:
        rename["longitude"] = "lon"
    return df.rename(columns=rename)


def _subset_copernicus(
    *,
    dataset_id: str,
    variables: Sequence[str],
    domain: Sequence[float],
    start_datetime: str,
    end_datetime: str,
    out_dir: Union[str, Path],
    out_name: str,
    dataset_version: Optional[str] = None,
    force: bool = False,
    start_depth: Optional[float] = None,
    end_depth: Optional[float] = None,
) -> Path:
    """Download a Copernicus Marine subset (cached)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    if out_path.exists() and not force:
        return out_path

    usr, psw = load_copernicus_creds()
    lat1, lon1, lat2, lon2 = map(float, domain)

    kwargs = dict(
        dataset_id=dataset_id,
        variables=list(variables),
        minimum_longitude=lon1,
        maximum_longitude=lon2,
        minimum_latitude=lat1,
        maximum_latitude=lat2,
        minimum_depth=start_depth,
        maximum_depth=end_depth,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        username=usr,
        password=psw,
        output_directory=str(out_dir),
        output_filename=out_path.name,
    )
    if dataset_version:
        kwargs["dataset_version"] = dataset_version

    copernicusmarine.subset(**kwargs)  # type: ignore[arg-type]
    return out_path


def _daily_mean_over_time(df: pd.DataFrame, value_cols: Sequence[str]) -> pd.DataFrame:
    """If a `time` column exists, average over time for each lat/lon."""
    if df.empty:
        return df
    if "time" not in df.columns:
        return df

    present = [c for c in value_cols if c in df.columns]
    if not present:
        return df

    return (
        df.groupby(["lat", "lon"], as_index=False)[present]
        .mean(numeric_only=True)
        .sort_values(["lat", "lon"], ignore_index=True)
    )


def get_altimetry(
    dt: Union[date, datetime],
    domain: Sequence[float],
    data_dir: Union[str, Path],
    *,
    prefix: str = "altimetry_nrt",
    dataset_version: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Daily altimetry subset: SLA + surface geostrophic currents.
    https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_NRT_008_046/description


    Returns columns: lat, lon, sla, ugos, vgos, current
    """
    day = _as_date(dt)
    start = f"{_ymd(day)}T00:00:00"
    end = start  # P1D product

    out_dir = Path(data_dir) / prefix
    out_name = f"{prefix}_{_ymd_compact(day)}.nc"

    try:
        nc_path = _subset_copernicus(
            dataset_id="cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
            variables=["sla", "ugos", "vgos"],
            domain=domain,
            start_datetime=start,
            end_datetime=end,
            out_dir=out_dir,
            out_name=out_name,
            dataset_version=dataset_version,
            force=force,
        )
        df = netcdf_to_dataframe(nc_path, variables=["sla", "ugos", "vgos"])
        if not df.empty and all(c in df.columns for c in ("ugos", "vgos")):
            df["current"] = (df["ugos"] ** 2 + df["vgos"] ** 2) ** 0.5
        return df
    except Exception:
        LOG.exception("Downloading %s for %s failed.", prefix, day)
        return pd.DataFrame()


def get_wind(
    dt: Union[date, datetime],
    domain: Sequence[float],
    data_dir: Union[str, Path],
    *,
    prefix: str = "wind_nrt",
    dataset_version: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Hourly wind subset aggregated to daily mean.
    https://data.marine.copernicus.eu/product/WIND_GLO_PHY_L4_NRT_012_004/description

    Returns columns: lat, lon, eastward_wind, northward_wind, wind_speed
    """
    day = _as_date(dt)

    start = f"{_ymd(day)}T00:00:00"
    end = f"{_ymd(day + timedelta(days=1))}T00:00:00"

    out_dir = Path(data_dir) / prefix
    out_name = f"{prefix}_{_ymd_compact(day)}.nc"

    try:
        nc_path = _subset_copernicus(
            dataset_id="cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
            variables=["eastward_wind", "northward_wind"],
            domain=domain,
            start_datetime=start,
            end_datetime=end,
            out_dir=out_dir,
            out_name=out_name,
            dataset_version=dataset_version,
            force=force,
        )
        df = netcdf_to_dataframe(nc_path, variables=["eastward_wind", "northward_wind"])
        df = _daily_mean_over_time(df, ["eastward_wind", "northward_wind"])
        if not df.empty and all(c in df.columns for c in ("eastward_wind", "northward_wind")):
            df["wind_speed"] = (df["eastward_wind"] ** 2 + df["northward_wind"] ** 2) ** 0.5
        return df
    except Exception:
        LOG.exception("Downloading %s for %s failed.", prefix, day)
        return pd.DataFrame()


def get_biogeochem(
    dt: Union[date, datetime],
    domain: Sequence[float],
    data_dir: Union[str, Path],
    prefix: str = "biogeochem",
    dataset_version: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Daily Biogeochem subset: Fe, NO3, PO4, Si.
    https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_BGC_001_028/description

    Returns columns: lat, lon, fe, no3, po4, si
    """
    day = _as_date(dt)
    start = f"{_ymd(day)}T00:00:00"
    end = start  # P1D product

    out_dir = Path(data_dir) / prefix
    out_name = f"{prefix}_{_ymd_compact(day)}.nc"

    try:
        nc_path = _subset_copernicus(
            dataset_id="cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m",
            variables=["fe", "no3", "po4", "si"],
            domain=domain,
            start_depth=0,
            end_depth=0.5,
            start_datetime=start,
            end_datetime=end,
            out_dir=out_dir,
            out_name=out_name,
            dataset_version=dataset_version,
            force=force,
        )        
        df = netcdf_to_dataframe(nc_path, variables=["fe", "no3", "po4", "si"])
        return df
    except Exception:
        LOG.exception("Downloading %s for %s failed.", prefix, day)
        return pd.DataFrame()

