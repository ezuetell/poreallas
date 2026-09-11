### Utilities to compute age-weighted mortality impacts relative to a base period
# ### Emily Zuetell
### July 7, 2026

import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import os
from dotenv import load_dotenv

import cil_regionalization as cilreg
from cil_regionalization.config import SourceUnitPolicies


from dataclasses import dataclass, field

@dataclass
class ImpactConfig:
    version: str 
    baseline_period: slice
    socioeconomics: object = None    # xr.Dataset; loaded in __post_init__ if None
    polygons: object = None          # geopandas.GeoDataFrame; loaded in __post_init__ if None
    hotonly: str = "net"
    rate: bool = False
    age_weight: bool = True
    cohort: str = "age65plus"
    dims: list = field(default_factory=lambda: ["number", "sample"])
    months: list = field(default_factory=lambda: [8, 9, 10, 11, 12, 1])

    def __post_init__(self):
        if self.polygons is None or self.socioeconomics is None:
            load_dotenv()
            data_dir = os.environ["DATA_DIR"]

        if self.polygons is None:
            polygons_uri = os.environ["POREALLAS_REGIONS_POLYGONS_URI"]
            self.polygons = (
                gpd.read_parquet(os.path.join(data_dir, polygons_uri))
                .rename(columns={"hierid": "region"})
                .set_index("region")
                .set_crs(epsg=4326)  # Assuming the data is WGS-84.
            )

        if self.socioeconomics is None:
            socio_uri = os.environ["POREALLAS_SOCIOECONOMICS_URI"]
            self.socioeconomics = xr.open_zarr(os.path.join(data_dir, socio_uri)).sel(year=2026)[
                ["pop0to4", "pop5to64", "pop65plus", "pop", "gdppc", "iso3"]
            ]

    def _pop_weight_sum(self, da, impact=True):
        if self.age_weight:
            pop_weight = xr.concat(
                [self.socioeconomics["pop0to4"], self.socioeconomics["pop5to64"], self.socioeconomics["pop65plus"]],
                dim=pd.Index(["age0to4", "age5to64", "age65plus"], name="age_cohort"),
            )
            age_weighted_total = da * pop_weight / 100000
            if self.rate:
                age_weighted_total = age_weighted_total * 100000 / self.socioeconomics["pop"]
            regional_sum = age_weighted_total.sum(dim="age_cohort")
            regional_sum.name = "age_weighted_impact" if impact else "age_weighted_effect"
        else:
            regional_sum = da.sel(age_cohort=self.cohort)
            if not self.rate:
                _col = f"pop{self.cohort[3:]}"
                regional_sum = da.sel(age_cohort=self.cohort) / 100000 * self.socioeconomics[_col]
            regional_sum.name = f"{self.cohort}_impact" if impact else f"{self.cohort}_effect"
        return regional_sum

    def compute_impact(self, projected, chunks={"number": -1, "sample": -1, "region": "auto"}, ensemble=False):
        valid_chunks = {k: v for k, v in chunks.items() if k in projected["/forecast_hotonly"]["effect"].dims}

        valid_terms = ["net", "hotonly", "coldonly"]
        if self.hotonly not in valid_terms:
            raise ValueError(f"Invalid term: {self.hotonly!r}. Must be one of {valid_terms}")

        group = {"net": "", "hotonly": "_hotonly", "coldonly": "_coldonly"}[self.hotonly]
        _baseline = (
            projected[f"/baseline{group}"]["effect"]
            .chunk({"region": "auto"})
            .sel(time=self.baseline_period)
            .groupby("time.month")
            .mean()
        )
        _forecast = projected[f"/forecast{group}"]["effect"].chunk(valid_chunks)
        if not ensemble:
            _forecast = _forecast.mean(dim="number")
        _forecast = _forecast.groupby("time.month").mean()

        impact = _forecast - _baseline
        impact.name = "impact"
        impact.attrs["long_name"] = "Temperature mortality impact"
        impact.attrs["units"] = "Deaths per 100,000 people" if self.rate else "Deaths"
        impact.attrs["hotonly"] = self.hotonly

        return self._pop_weight_sum(impact)


def get_baseline_period(effect_xr, years=30):
    # Baseline
    _max_year = effect_xr["/baseline"]["time"].max().dt.year.item()
    # Subtracting 29 even though we want 30 year baseline because the time slice is an inclusive range.
    _min_year = _max_year - (years - 1)
    baseline_period = slice(str(_min_year), str(_max_year))
    return baseline_period



def compute_global_impact(impact, socioeconomics, rate, group_dim="region"):
    if rate:
        pop = socioeconomics["population"].sel(region=impact.region)
        return (impact * pop).sum(dim=group_dim) / pop.sum(dim=group_dim)
    return impact.sum(dim=group_dim)

### Analysis Functions ###


def xarray_to_gpd(data, polygons, crs="ESRI:54030"):
    _polygons_data = polygons.merge(
        data.to_dataframe(name=data.name or "value").reset_index(),
        on="region",
    )

    _polygons_data = _polygons_data.cx[:, -60:90].to_crs(crs)

    return _polygons_data


def compute_stats(da, dim="number"):
    """
    From an xarray, return statistics along dimension, 'dim' as an xr.Dataset.
    """
    mean = da.mean(dim=dim)
    std = da.std(dim=dim)
    min = da.min(dim=dim)
    max = da.max(dim=dim)
    p = da.quantile([0.10, 0.17, 0.5, 0.83, 0.9], dim=dim)

    p10 = p.sel(quantile=0.10, drop=True)
    p17 = p.sel(quantile=0.17, drop=True)
    p50 = p.sel(quantile=0.50, drop=True)
    p83 = p.sel(quantile=0.83, drop=True)
    p90 = p.sel(quantile=0.90, drop=True)
    likely_range = p83 - p17

    return xr.Dataset(
        {
            "median": p50,
            "p17": p17,
            "p83": p83,
            "likely_range_IPCC": likely_range,
            "mean": mean,
            "std": std,
            "min": min,
            "max": max,
            "p10": p10,
            "p90": p90,
        }
    )
def merge_polygon_stats(stats_ds, polygon, merge_key="region"):
    """Merge a stats Dataset onto a polygon GeoDataFrame on merge_key."""
    return polygon.merge(stats_ds.to_dataframe().reset_index(), on=merge_key)

### Regionalization Functions ###
def redistribute_adm1(impact, config, weights, chunks=None):
    if chunks is None:
        chunks = {dim: "auto" for dim in config.dims}
    impact = impact.chunk(chunks)

    df = impact.to_dask_dataframe(dim_order=None).reset_index()
    df = df.rename(columns={"region": "hierid"})

    col_regions = {(h,) for h in df["hierid"].unique().compute()}
    kind = "intensive" if config.rate else "extensive"

    # If apply_weights requires pandas, materialize just before the call:
    adm1 = cilreg.apply_weights(
        weights,
        df.compute(),  # drop this line if apply_weights accepts dask directly
        kind=kind,
        weight="pop",
        value_col="value",
        data_version="world-combo-201710",
        restrict_to_sources=col_regions,
        allow_partial_coverage=True,
        policies=SourceUnitPolicies(on_unmatched="skip", on_zero_weight="skip"),
    ).frame

    index_cols = [c for c in adm1.columns if c != "value"]
    return adm1.set_index(index_cols).to_xarray()["value"]

def aggregate_by_iso(ds, polygon, operation="sum"):
    iso_map = polygon["ISO"].reindex(ds["region"].values)
    ds = ds.assign_coords(ISO=("region", iso_map.to_numpy(dtype=object)))

    grouped = ds.groupby("ISO")
    if operation == "sum":
        ds = grouped.sum(dim="region")
    elif operation == "mean":
        ds = grouped.mean(dim="region")
    else:
        raise ValueError(f"Unsupported operation: {operation!r}")

    polygon = polygon.copy()
    polygon["geometry"] = polygon["geometry"].buffer(0)
    polygon = polygon.dissolve(by="ISO", aggfunc="first")

    return ds, polygon

def aggregate_impact(impact, config, group_level):
    if group_level == "IR":
        return impact, "region", ["region", "ISO"]

    if group_level == "ISO":
        if config.rate:
            raise ValueError("ISO grouping is not supported when rate=True")
        impact, _ = aggregate_by_iso(impact, config.polygons, operation="sum")
        return impact, "ISO", ["ISO"]

    if group_level == "ADM1":
        weights = cilreg.fetch_weights(
            "gadm41-adm1-per-destination" if config.rate else "gadm41-adm1-per-source"
        )
        impact = redistribute_adm1(impact, config, weights)
        return impact, "GID_1", ["GID_0", "GID_1"]

    raise ValueError(f"Unsupported group_level: {group_level!r}")

###Output Functions ###
def dataset_to_dataframe(ds):
    if len(ds.dims) == 0:
        return pd.DataFrame({k: [v.values.item()] for k, v in ds.data_vars.items()})
    return ds.to_dataframe().reset_index()

def make_csv(
    effect,
    config: ImpactConfig,
    group_level="IR",
    filename_template="{version}_{hotonly}_{scope}_{rate_l}_{stat_scope}_{group_level}.csv",
    output_scope=["regional_monthly", "regional_6mo", "global_monthly", "global_6mo"],
):
    rate_l = "rate" if config.rate else "total"

    impact = config.compute_impact(effect.chunk({dim: -1 for dim in config.dims}), ensemble=True)
    impact = impact.sel(month=config.months)

    polygon = config.polygons
    impact, merge_key, base_cols = aggregate_impact(impact, config, group_level)

    stat_cols = ["median", "p17", "p83", "likely_range_IPCC", "mean", "std", "min", "max", "p10", "p90"]

    if "regional_monthly" in output_scope:
        _polygons_impact = dataset_to_dataframe(compute_stats(impact, dim=config.dims))
        wide = _polygons_impact.pivot(
            index=base_cols,
            columns="month", values=stat_cols,
        )
        wide.columns = [f"month {m} {stat}" for stat, m in wide.columns]
        stat_col_names = wide.columns.difference(base_cols)
        wide[stat_col_names] = wide[stat_col_names].round(0).astype("Int64")
        wide = wide.reset_index()
        wide.to_csv(
            filename_template.format(
                version=config.version,
                hotonly=config.hotonly,
                rate_l=rate_l,
                scope="all",
                stat_scope="",
                group_level=group_level,
            ),
            index=False,
        )

    if "regional_6mo" in output_scope:
        mo6 = impact.sum(dim="month")
        _polygons_mo6 = dataset_to_dataframe(compute_stats(mo6, dim=config.dims))
        base_cols = ["ISO"] if merge_key == "ISO" else ["region", "ISO"]
        mo6_out = _polygons_mo6[base_cols + stat_cols]
        stat_col_names = mo6_out.columns.difference(base_cols)
        mo6_out[stat_col_names] = mo6_out[stat_col_names].round(0).astype("Int64")
        mo6_out.to_csv(
            filename_template.format(
                version=config.version,
                hotonly=config.hotonly,
                rate_l=rate_l,
                scope="6mo",
                stat_scope="",
                group_level=group_level,
            ),
            index=False,
        )

    if "global_monthly" in output_scope or "global_6mo" in output_scope:
        global_impact = compute_global_impact(impact, config.socioeconomics, rate=config.rate, group_dim=merge_key)

        if "global_monthly" in output_scope:
            global_monthly = dataset_to_dataframe(compute_stats(global_impact, dim=config.dims))
            stat_col_names = global_monthly.columns.difference(["region", "ISO"])
            global_monthly[stat_col_names] = global_monthly[stat_col_names].round(0).astype("Int64")
            global_monthly.to_csv(
                filename_template.format(
                    version=config.version,
                    hotonly=config.hotonly,
                    rate_l=rate_l,
                    scope="monthly_global",
                    stat_scope="",
                    group_level=group_level,
                ),
                index=False,
            )
        if "global_6mo" in output_scope:
            global_mo6 = dataset_to_dataframe(compute_stats(global_impact.sum(dim="month"), dim=config.dims))
            stat_col_names = global_mo6.columns.difference(["region", "ISO"])
            global_mo6[stat_col_names] = global_mo6[stat_col_names].round(0).astype("Int64")
            global_mo6.to_csv(
            filename_template.format(
                version=config.version,
                hotonly=config.hotonly,
                rate_l=rate_l,
                scope="6mo_global",
                stat_scope="",
                group_level=group_level,
            ),
                index=False,
            )


def build_stats_text(da, dim=None, fmt="{:.2f}"):
    stats = {
        "mean": float(da.mean(dim=dim)),
        "std": float(da.std(dim=dim)),
        "min": float(da.min(dim=dim)),
        "max": float(da.max(dim=dim)),
    }
    return "\n".join(f"{k}: {fmt.format(v)}" for k, v in stats.items())


def add_stats_annotation(text, ax, loc="upper left"):
    loc_map = {
        "upper right": (0.975, 1.05, "top", "right"),
        "upper left": (0.005, 1.05, "top", "left"),
        "lower right": (0.975, 0.025, "bottom", "right"),
        "lower left": (0.025, 0.025, "bottom", "left"),
    }
    x, y, va, ha = loc_map[loc]

    return ax.annotate(
        text,
        xy=(x, y),
        xycoords="axes fraction",
        ha=ha,
        va=va,
        fontsize=9,
    )


##### Plotting Functions #####
from functools import lru_cache
import regionmask


# Get and store land data
@lru_cache(maxsize=None)
def _get_land(crs):
    land = gpd.read_file(geodatasets.get_path("naturalearth land"))
    return land.cx[:, -60:90].to_crs(crs)


@lru_cache(maxsize=None)
def _get_land_mask(lon_key, lat_key):
    dummy = xr.DataArray(
        np.zeros((len(lat_key), len(lon_key))),
        coords={"lat": list(lat_key), "lon": list(lon_key)},
        dims=["lat", "lon"],
    )
    return regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(dummy)


def compute_area_weighted_mean(ds, lat_name="lat", lon_name="lon"):
    weights = np.cos(np.deg2rad(ds[lat_name]))
    weights.name = "weights"
    return ds.weighted(weights).mean((lat_name, lon_name))

def land_only(da, lat_name="lat", lon_name="lon"):
    da = da.rename({lon_name: "lon", lat_name: "lat"})
    mask = _get_land_mask(tuple(da.lon.values), tuple(da.lat.values))
    return da.where(mask.notnull() & (da.lat > -60))


def get_step(target_range, min_bins=2, max_bins=8):
    magnitude = 10 ** np.floor(np.log10(target_range))
    for _ in range(3):
        for step in [1, 2, 5, 10]:
            candidate = step * magnitude
            n_bins = target_range / candidate
            if min_bins <= n_bins <= max_bins:
                return candidate
        magnitude /= 10
    return magnitude * 10


def nice_step(target):
    """Snap an arbitrary step size to the nearest round value (1, 2, 5, or 10 x 10**n)."""
    magnitude = 10 ** np.floor(np.log10(target))
    options = np.array([1, 2, 5, 10]) * magnitude
    return options[np.argmin(np.abs(options - target))]


def round_bounds(bounds, step):
    """Remove floating-point drift so bounds land exactly on multiples of step/2."""
    return np.round(bounds * 2 / step) * step / 2


def get_ticks(bounds, step=None, max_ticks=10, symmetric=False):
    if symmetric:
        absmax = bounds.max()
        half = step * np.arange(0, int(np.floor(absmax / step)) + 1)
        stride = int(np.ceil((2 * len(half) - 1) / max_ticks))
        half = half[::stride]
        return np.unique(np.concatenate([-half, half]))
    else:
        stride = int(np.ceil(len(bounds) / max_ticks))
        return bounds[::stride]


def make_cmap(bounds, cm="bwr"):
    cmap_base = plt.get_cmap(cm)
    colors = cmap_base(np.linspace(0, 1, len(bounds) - 1))

    diverging_cmaps = {
        "bwr",
        "seismic",
        "coolwarm",
        "RdBu",
        "RdYlBu",
        "PiYG",
        "PRGn",
        "BrBG",
    }
    base_name = cm[:-2] if cm.endswith("_r") else cm
    if base_name in diverging_cmaps:
        center_idx = len(colors) // 2
        colors[center_idx] = [0.95, 0.95, 0.95, 1]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    return cmap, norm, sm


def build_colormap(gdf=None, col=None, cm="bwr", vmin=None, vmax=None, n_colors=None):
    # Build discrete colormap, branching on diverging vs sequential
    diverging_cmaps = {
        "bwr",
        "seismic",
        "coolwarm",
        "RdBu",
        "RdYlBu",
        "PiYG",
        "PRGn",
        "BrBG",
    }
    base_name = cm[:-2] if cm.endswith("_r") else cm

    if base_name in diverging_cmaps:
        # Symmetric around zero, with a white center band
        if vmin is not None or vmax is not None:
            absmax = max(abs(vmin), abs(vmax))
        else:
            absmax = math.ceil(gdf[col].abs().quantile(0.95))

        step = nice_step((2 * absmax) / n_colors) if n_colors else get_step(2 * absmax)
        bounds = np.arange(
            -np.ceil(absmax / step) * step - step / 2,
            np.ceil(absmax / step) * step + step,
            step,
        )
        bounds = round_bounds(bounds, step)
        cmap, norm, sm = make_cmap(bounds, cm=cm)
    else:
        # Sequential, uses vmin/vmax (or data min/max) directly, no center-forcing
        if vmin is not None and vmax is not None:
            lo, hi = vmin, vmax
        else:
            lo = vmin if vmin is not None else gdf[col].min()
            hi = vmax if vmax is not None else gdf[col].max()

        step = nice_step((hi - lo) / n_colors) if n_colors else get_step(hi - lo)
        bounds = np.arange(
            np.floor(lo / step) * step, np.ceil(hi / step) * step + step, step
        )
        bounds = round_bounds(bounds, step)
        cmap = plt.get_cmap(cm, len(bounds) - 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    ticks = get_ticks(bounds, step=step, symmetric=(base_name in diverging_cmaps))
    return cmap, norm, sm, ticks, step


import math
import geodatasets

land = gpd.read_file(geodatasets.get_path("naturalearth land"))


def plot_single(
    gdf,
    col,
    sup_title="",
    save_title="",
    cm="bwr",
    cbar_label=None,
    vmin=None,
    vmax=None,
    alpha = 1,
    edgecolor=None,
    linewidth=0,
    ax=None,
    cbar_location="right",
    colorbar=True,
    n_colors=None,
    annotation=None,
    target_crs="ESRI:54030",
):

    if gdf.crs is None:
        raise ValueError("gdf has no CRS set")
    if gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)

    land = _get_land(target_crs)

    cmap, norm, sm, ticks, step = build_colormap(
        gdf, col, cm=cm, vmin=vmin, vmax=vmax, n_colors=n_colors
    )

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    # Plot background land
    land.plot(color="lightgray", edgecolor="lightgray", ax=ax)
    # Plot data
    gdf.plot(
        column=col,
        legend=False,
        ax=ax,
        cmap=cmap,
        norm=norm,
        alpha = alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.set_axis_off()

    if colorbar:
        orientation = "horizontal" if cbar_location in ("bottom", "top") else "vertical"
        cb = fig.colorbar(
            sm,
            ax=ax,
            location=cbar_location,
            orientation=orientation,
            shrink=0.6,
            ticks=ticks,
            label=cbar_label,
        )
        decimals = max(0, int(np.ceil(-np.log10(step)))) if step < 1 else 0
        cb.set_ticklabels([f"{t:.{decimals}f}" for t in ticks])
        for label in cb.ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    if annotation:
        add_stats_annotation(annotation, ax)

    if standalone and save_title:
        fig.savefig(save_title, dpi=600, bbox_inches="tight")
    return ax


def plot_monthly(
    gdf,
    col,
    sup_title="",
    save_title="",
    cm="bwr",
    cbar_label=None,
    vmin=None,
    vmax=None,
    edgecolor=None,
    n_colors=None,
    linewidth=0,
    month_order=None,
):
    if vmin is None:
        vmin = -gdf[col].abs().quantile(0.95)
    if vmax is None:
        vmax = gdf[col].abs().quantile(0.95)

    if month_order is None:
        months = sorted(gdf["month"].unique())
    else:
        present = set(gdf["month"].unique())
        months = [m for m in month_order if m in present]

    fig, axes = plt.subplots(2, 3, figsize=(16, 6))

    for ax, month in zip(axes.flat, months):
        group = gdf[gdf["month"] == month]
        plot_single(
            group,
            col,
            cm=cm,
            vmin=vmin,
            vmax=vmax,
            edgecolor=edgecolor,
            linewidth=linewidth,
            ax=ax,
            colorbar=False,
            n_colors=n_colors,
        )
        ax.set_title(f"Month {month}")

    for ax in axes.flat[len(months) :]:
        ax.set_axis_off()

    cmap, norm, sm, ticks, step = build_colormap(
        gdf, col, cm=cm, vmin=vmin, vmax=vmax, n_colors=n_colors
    )
    fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        location="right",
        shrink=0.6,
        ticks=ticks,
        label=cbar_label,
    )

    fig.suptitle(sup_title, fontsize=14)
    fig.savefig(save_title, dpi=600, bbox_inches="tight")
    return fig


def plot_aggregate(
    gdf,
    col,
    agg="sum",
    rate=True,
    sup_title="",
    save_title="",
    cm="bwr",
    cbar_label=None,
    vmin=None,
    vmax=None,
    edgecolor=None,
    linewidth=0,
    ax=None,
    cbar_location="right",
    colorbar=True,
    n_colors=None,
    annotation=None,
    target_crs="ESRI:54030",
):

    annual = gdf.groupby(["region", "geometry"], as_index=False)[col].agg(agg)
    annual = gpd.GeoDataFrame(annual, geometry="geometry", crs=gdf.crs)

    ax = plot_single(
        annual,
        col,
        cm=cm,
        ax=ax,
        cbar_label=cbar_label,
        vmin=vmin,
        vmax=vmax,
        edgecolor=edgecolor,
        linewidth=linewidth,
        cbar_location=cbar_location,
        colorbar=colorbar,
        n_colors=n_colors,
        annotation=annotation,
        target_crs=target_crs,
    )

    fig = ax.figure
    fig.suptitle(sup_title, fontsize=14)
    fig.savefig(save_title, dpi=600, bbox_inches="tight")
    return fig
