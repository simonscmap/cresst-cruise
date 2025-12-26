from __future__ import annotations

"""Cartopy-based plotting helpers for CRESST daily maps.

This module expects *long-form* dataframes with columns:

- lat, lon
- one or more data variables (e.g., sst, chlor_a, sla)

It produces...
"""

import calendar
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cmocean
except Exception:  # pragma: no cover
    cmocean = None  # type: ignore

from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm


# ---------------------------- Data reshaping ---------------------------- #


def field_to_grid(field: pd.DataFrame, variable: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a long-form dataframe into a regular lat/lon grid.

    Returns
    -------
    lat (1D), lon (1D), data (2D) with shape (nlat, nlon)
    """
    if field.empty:
        raise ValueError("Empty dataframe")
    if not {"lat", "lon", variable}.issubset(field.columns):
        missing = {"lat", "lon", variable} - set(field.columns)
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    piv = (
        field.pivot(index="lat", columns="lon", values=variable)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    lat = piv.index.to_numpy()
    lon = piv.columns.to_numpy()
    data = piv.to_numpy()
    return lat, lon, data


def vector_to_grid(
    field: pd.DataFrame,
    u_name: str,
    v_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert vector components in a long-form dataframe to 2D grids."""
    lat, lon, u = field_to_grid(field, u_name)
    _, _, v = field_to_grid(field, v_name)
    return lat, lon, u, v


# ---------------------------- Station overlay ---------------------------- #


def load_stations(csv_path: Union[str, Path] = "stations.csv") -> Optional[pd.DataFrame]:
    """Load station lat/lon from a CSV with columns: lat, lon."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None

    stations = pd.read_csv(csv_path)
    required = {"lat", "lon"}
    if not required.issubset(stations.columns):
        return None

    # Make sure numeric
    stations = stations.copy()
    stations["lat"] = pd.to_numeric(stations["lat"], errors="coerce")
    stations["lon"] = pd.to_numeric(stations["lon"], errors="coerce")
    stations = stations.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return stations if len(stations) else None


def overlay_stations(stations: Optional[pd.DataFrame], ax, *, zorder: int = 5) -> None:
    """Overlay station markers on a Cartopy axes."""
    if stations is None or stations.empty:
        return

    transform = ccrs.PlateCarree()

    # Halo + dot gives readability on dark/light backgrounds
    ax.scatter(
        stations["lon"],
        stations["lat"],
        s=60,
        marker="o",
        facecolors="none",
        edgecolors="white",
        linewidths=2.0,
        zorder=zorder,
        transform=transform,
    )
    ax.scatter(
        stations["lon"],
        stations["lat"],
        s=18,
        marker="o",
        c="black",
        linewidths=0,
        zorder=zorder + 1,
        transform=transform,
    )


# ---------------------------- Map styling ---------------------------- #


@dataclass(frozen=True)
class VarStyle:
    cmap: str
    log: bool = False
    diverging_center: Optional[float] = None


def _default_cmap(name: str) -> str:
    lname = name.lower()

    def _pick_registered(candidates: list[str], default: str) -> str:
        try:
            import matplotlib as mpl

            registered = set(mpl.colormaps)
        except Exception:
            try:
                registered = set(plt.colormaps())
            except Exception:
                registered = set()
        for c in candidates:
            if c in registered:
                return c
        return default

    if cmocean is not None:
        if "sst" in lname:
            return _pick_registered(["RdYlBu_r", "cmo.thermal", "thermal"], "RdYlBu_r")
        if "chlor" in lname or "chl" in lname:
            return _pick_registered(["cmo.algae", "algae"], "viridis")
        if "sla" in lname or "adt" in lname:
            return _pick_registered(["cmo.balance", "balance"], "bwr")
        if "wind" in lname:
            return _pick_registered(["cmo.matter", "matter"], "viridis")
        if "current" in lname or "ugos" in lname or "vgos" in lname:
            return _pick_registered(["bone", "speed"], "viridis")
        if "par" in lname:
            return _pick_registered(["cmo.solar", "solar"], "viridis")

    return "viridis"


def style_for_variable(variable: str) -> VarStyle:
    lname = variable.lower()

    # Chlorophyll is commonly viewed on log scale.
    if lname in {"chlor_a", "chl", "chl_a", "chlor"} or "chlor" in lname or "chl" in lname:
        return VarStyle(cmap=_default_cmap(variable), log=True)

    # SLA is naturally centered at 0.
    if "sla" in lname:
        return VarStyle(cmap=_default_cmap(variable), diverging_center=0.0)

    return VarStyle(cmap=_default_cmap(variable))


def time_label(time_like: object) -> str:
    if isinstance(time_like, int):
        return calendar.month_name[time_like]
    return str(time_like)


def make_geo_axes(
    extent: Sequence[float],
    *,
    figsize: Tuple[float, float] = (8.5, 5.5),
    land_scale: str = "10m",
    show_borders: bool = False,
):
    """Create a Cartopy PlateCarree axes with ocean/land + labeled gridlines."""
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=proj)

    lon1, lon2, lat1, lat2 = extent
    ax.set_extent([lon1, lon2, lat1, lat2], crs=proj)
    land = cfeature.LAND.with_scale(land_scale)
    coastline = cfeature.COASTLINE.with_scale(land_scale)

    # Avoid Cartopy's OCEAN feature here (it can trigger Shapely warnings on some setups).
    # Instead, set the axes background and draw land + coastlines.
    ax.set_facecolor("white")
    ax.add_feature(land, facecolor="lightgray", zorder=1)
    ax.add_feature(coastline, linewidth=0.8, zorder=2)

    if show_borders:
        borders = cfeature.BORDERS.with_scale(land_scale)
        ax.add_feature(borders, linewidth=0.5, linestyle=":", zorder=2)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", alpha=0.6)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    return fig, ax


def _robust_limits(data: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> Tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.nanpercentile(finite, p_lo)), float(np.nanpercentile(finite, p_hi))


def _make_norm(data: np.ndarray, style: VarStyle) -> Normalize:
    if style.diverging_center is not None:
        # symmetric bounds around center
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return TwoSlopeNorm(vcenter=style.diverging_center, vmin=-1, vmax=1)
        amp = float(np.nanpercentile(np.abs(finite - style.diverging_center), 98))
        vmin = style.diverging_center - amp
        vmax = style.diverging_center + amp
        return TwoSlopeNorm(vcenter=style.diverging_center, vmin=vmin, vmax=vmax)

    vmin, vmax = _robust_limits(data)

    if style.log:
        # LogNorm requires positive values; compute limits on positive subset.
        pos = data[np.isfinite(data) & (data > 0)]
        if pos.size == 0:
            return Normalize(vmin=vmin, vmax=vmax)
        vmin = float(np.nanpercentile(pos, 2))
        vmax = float(np.nanpercentile(pos, 98))
        return LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, 1e-12))

    return Normalize(vmin=vmin, vmax=vmax)


# ---------------------------- Scalar plots ---------------------------- #


def plot_map(
    lat: np.ndarray,
    lon: np.ndarray,
    data: np.ndarray,
    *,
    variable: str,
    unit: str,
    time_like: object,
    save_path: Union[str, Path],
    stations: Optional[pd.DataFrame] = None,
    source: Optional[str] = None,
) -> None:
    extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
    fig, ax = make_geo_axes(extent)

    Lon2, Lat2 = np.meshgrid(lon, lat)
    style = style_for_variable(variable)
    norm = _make_norm(data, style)

    im = ax.pcolormesh(
        Lon2,
        Lat2,
        data,
        transform=ccrs.PlateCarree(),
        cmap=style.cmap,
        norm=norm,
        shading="auto"
    )

    overlay_stations(stations, ax)

    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.92)
    label_unit = unit.strip() if unit else ""
    cbar.set_label(f"{variable} ({label_unit})" if label_unit else variable)

    ax.set_title(f"{variable} — {time_label(time_like)}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------- Gradients ---------------------------- #


def compute_gradient(lat: np.ndarray, lon: np.ndarray, data: np.ndarray):
    """Compute horizontal gradient of a scalar field on a lat/lon grid.

    Returns
    -------
    dvar_dx, dvar_dy in units per meter; grad_mag_km in units per km.
    """
    R = 6_371_000.0  # meters

    dlat_deg = float(np.diff(lat).mean())
    dlon_deg = float(np.diff(lon).mean())

    phi0 = np.deg2rad(float(lat.mean()))
    dy = np.deg2rad(dlat_deg) * R
    dx = np.deg2rad(dlon_deg) * R * np.cos(phi0)

    dvar_dy, dvar_dx = np.gradient(data, dy, dx)

    grad_mag_km = np.hypot(dvar_dx, dvar_dy) * 1000.0
    return dvar_dx, dvar_dy, grad_mag_km


def plot_gradient_mag(
    lat: np.ndarray,
    lon: np.ndarray,
    grad_mag_km: np.ndarray,
    *,
    variable: str,
    unit: str,
    time_like: object,
    save_path: Union[str, Path],
    stations: Optional[pd.DataFrame] = None
) -> None:
    extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
    fig, ax = make_geo_axes(extent)

    Lon2, Lat2 = np.meshgrid(lon, lat)
    vmin, vmax = _robust_limits(grad_mag_km)
    # Use a Colormap object (not a string name) to avoid registration/name differences
    # across Matplotlib/cmocean installations.
    if cmocean is not None:
        try:
            cmap = plt.get_cmap("cmo.amp")
        except Exception:
            cmap = cmocean.cm.amp
    else:
        cmap = "magma"

    im = ax.pcolormesh(
        Lon2,
        Lat2,
        grad_mag_km,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        shading="auto"
    )

    overlay_stations(stations, ax)

    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.92)
    label_unit = unit.strip() if unit else ""
    cbar.set_label(f"|∇{variable}| ({label_unit}/km)" if label_unit else f"|∇{variable}| (/km)")

    ax.set_title(f"Gradient magnitude: {variable} — {time_label(time_like)}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------- Vector plots ---------------------------- #


def plot_vector_field(
    field: pd.DataFrame,
    *,
    u_name: str,
    v_name: str,
    speed_name: str,
    unit: str,
    time_like: object,
    save_path: Union[str, Path],
    stations: Optional[pd.DataFrame] = None,
    max_arrows: int = 30
) -> None:
    """Plot a vector field (quiver) over a scalar speed background."""
    lat, lon, u, v = vector_to_grid(field, u_name, v_name)
    _, _, speed = field_to_grid(field, speed_name)

    extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
    fig, ax = make_geo_axes(extent)

    Lon2, Lat2 = np.meshgrid(lon, lat)

    style = style_for_variable(speed_name)
    norm = _make_norm(speed, style)

    im = ax.pcolormesh(
        Lon2,
        Lat2,
        speed,
        transform=ccrs.PlateCarree(),
        cmap=style.cmap,
        norm=norm,
        shading="auto",
        zorder=3,
    )

    # Choose a reasonable downsample for arrows.
    skip_lat = max(1, int(np.ceil(len(lat) / max_arrows)))
    skip_lon = max(1, int(np.ceil(len(lon) / max_arrows)))

    U = u[::skip_lat, ::skip_lon]
    V = v[::skip_lat, ::skip_lon]
    LON = Lon2[::skip_lat, ::skip_lon]
    LAT = Lat2[::skip_lat, ::skip_lon]

    # Dynamic scale: larger values -> higher scale -> shorter arrows.
    ref = float(np.nanpercentile(np.hypot(U, V), 90)) if np.isfinite(U).any() else 1.0
    scale = max(ref * 20, 1e-6)
    Q = ax.quiver(
        LON,
        LAT,
        U,
        V,
        transform=ccrs.PlateCarree(),
        pivot="mid",
        angles="uv",
        units="height",
        scale=scale,
        width=0.005,
        color="r" if speed_name.lower() == "current" else "k",
        alpha=0.7,
        zorder=4,
    )

    # Quiver key: show a representative speed
    key_val = 0.0
    for candidate in (1.0, 2.0, 5.0, 10.0):
        if ref >= candidate:
            key_val = candidate
    if key_val > 0:
        ax.quiverkey(Q, 0.92, 0.05, key_val, f"{key_val:g} {unit}", labelpos="E", coordinates="axes")

    overlay_stations(stations, ax)

    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.92)
    cbar.set_label(f"{speed_name} ({unit})")

    ax.set_title(f"{speed_name} + vectors — {time_label(time_like)}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------- Convenience wrappers ---------------------------- #


def plot_static_map(
    field: pd.DataFrame,
    variable: str,
    unit: str,
    time_like: object,
    fig_dir: Union[str, Path],
    stations: Optional[pd.DataFrame] = None,
    *,
    include_gradient: bool = True,
) -> None:
    """Plot a scalar map and (optionally) its gradient magnitude."""
    lat, lon, data = field_to_grid(field, variable)

    fig_dir = Path(fig_dir)
    plot_map(
        lat,
        lon,
        data,
        variable=variable,
        unit=unit,
        time_like=time_like,
        save_path=fig_dir / f"{variable}_{time_label(time_like)}.png",
        stations=stations,
    )

    if include_gradient:
        _, _, grad_mag_km = compute_gradient(lat, lon, data)
        plot_gradient_mag(
            lat,
            lon,
            grad_mag_km,
            variable=variable,
            unit=unit,
            time_like=time_like,
            save_path=fig_dir / f"grad_{variable}_{time_label(time_like)}.png",
            stations=stations,
        )
