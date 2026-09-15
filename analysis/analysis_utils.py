### Utilities to compute age-weighted mortality impacts relative to a base period
# ### Emily Zuetell
### July 7, 2026

import xarray as xr
import pandas as pd
import geopandas as gpd
import dask_geopandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from functools import lru_cache
import regionmask
import geodatasets

import os
from dotenv import load_dotenv

import cil_regionalization as cilreg
from cil_regionalization.config import SourceUnitPolicies


from dataclasses import dataclass, field

@dataclass
class ImpactConfig: # Class for impact computation
    version: str 
    baseline_period: slice
    polygons_path: str = None         # path to polygons parquet file
    socioeconomics_path: str = None   # path to socioeconomics zarr store
    regions_path: str = None          # segment weights
    socioeconomics: object = None    # xr.Dataset; loaded in __post_init__ if None
    polygons: object = None          # geopandas.GeoDataFrame; loaded in __post_init__ if None
    dims: list = None               # Dims preserved for uncertainty
    months: list = None             # Forecast Months (6-months) #### todo get rid of defaults
    hotonly: str = "net"             # "hotonly", "coldonly", or "net" 
    rate: bool = False               # "False" = Total Deaths, "True" = Mortality Rate
    age_weight: bool = True          # Age-cohort weighting
    cohort: str = "age65plus"


    def __post_init__(self):
        if self.polygons is None:
            self.polygons = (
                gpd.read_parquet(self.polygons_path)
                .rename(columns={"hierid": "region"})
                .set_index("region")
                .set_crs(epsg=4326)  # Assuming the data is WGS-84.
            )

        if self.socioeconomics is None:
            self.socioeconomics = xr.open_zarr(self.socioeconomics_path).sel(year=2026)[
                ["pop0to4", "pop5to64", "pop65plus", "pop", "gdppc", "iso3"]
            ]
## TODO Move these functions outside of class    
    def _pop_weight_sum(self, da, impact=True): # TODO two functions: rate and total
        """
        Parameters
        ----------
        da : xr.DataArray
            Mortality effect array (deaths per 100,000) with an 'age_cohort'
            dimension when self.age_weight is False.
        impact : bool, default True
            If True, names the output "*_impact"; if False, names it "*_effect".
            Distinguishes a computed impact (relative to baseline) from a raw
            effect value.

        Returns
        -------
        xr.DataArray
            Population-weighted mortality total or rate, named according to
            `impact` and the weighting strategy used.
        """
        def _pop_weight_total():
            if self.age_weight:
                # Age Weight = Mortality Rate * Pop Share * Total Pop/100k
                age_weighted_total = da * pop_weight * total_pop / 100000
                # Sum over age_cohort for each region (sum(pop_weight = 1))
                age_sum = age_weighted_total.sum(dim="age_cohort")
                age_sum.name = "age_weighted_impact" if impact else "age_weighted_effect"
            else:
                cohort_rate = da.sel(age_cohort=self.cohort)
                # Population of specified age cohort
                pop_col = f"pop{self.cohort[3:]}"
                #Total deaths = rate*pop/100k
                age_sum = cohort_rate * self.socioeconomics[pop_col] / 100000
                age_sum.name = f"{self.cohort}_impact" if impact else f"{self.cohort}_effect"
            return age_sum

        def _pop_weight_rate():
            if self.age_weight:
                #Age-weighted rate = sum(rate * popshare)
                age_weighted_rate = da * pop_weight
                age_sum = age_weighted_rate.sum(dim="age_cohort")
                age_sum.name = "age_weighted_impact" if impact else "age_weighted_effect"
            else:
                # Return selected age_cohort
                age_sum = da.sel(age_cohort=self.cohort)
                age_sum.name = f"{self.cohort}_impact" if impact else f"{self.cohort}_effect"
            return age_sum

        if self.age_weight:
            total_pop = self.socioeconomics["pop"]
            # Comput pop share (cohort_pop/total_pop)
            pop_weight = xr.concat(
                [self.socioeconomics["pop0to4"], self.socioeconomics["pop5to64"], self.socioeconomics["pop65plus"]],
                dim=pd.Index(["age0to4", "age5to64", "age65plus"], name="age_cohort"),
            ) / total_pop

        return _pop_weight_rate() if self.rate else _pop_weight_total()

    def compute_impact(self, projected, chunks={"number": -1, "sample": -1, "region": "auto"}, ensemble=False):
        """
        Parameters
        ----------
        projected : xarray.DataTree
            Datatree with "/baseline", "/forecast", and their "_hotonly"/
            "_coldonly" leaves, each holding an "effect" DataArray (deaths/100k) with
            a "time" dimension.
        chunks : dict, default {"number": -1, "sample": -1, "region": "auto"}
            Chunk sizes applied to the forecast effect array.
        ensemble : bool, default False
            If True, keep the "number" (ensemble) dimension in the forecast
            climatology. If False, average over "number" before computing
            impact.

        Returns
        -------
        xr.DataArray
            Population-weighted impact, as returned by self._pop_weight_sum.

        Raises
        ------
        ValueError
            If self.hotonly is not one of "net", "hotonly", or "coldonly".
        """
        # Check for valid chunks based on dims present
        valid_chunks = {k: v for k, v in chunks.items() if k in projected["/forecast_hotonly"]["effect"].dims}
        # Check validity of "hotonly" item
        valid_terms = ["net", "hotonly", "coldonly"]
        if self.hotonly not in valid_terms:
            raise ValueError(f"Invalid term: {self.hotonly!r}. Must be one of {valid_terms}")

        #Select leaf according to "hotonly" flag
        group = {"net": "", "hotonly": "_hotonly", "coldonly": "_coldonly"}[self.hotonly]

        # Monthly mean over baseline period
        _baseline = (
            projected[f"/baseline{group}"]["effect"]
            .chunk({"region": "auto"})
            .sel(time=self.baseline_period) #Select configured baseline period [Should typically be 30yrs]
            .groupby("time.month") #Group all years by month
            .mean() #Monthly average
        )
        _forecast = projected[f"/forecast{group}"]["effect"].chunk(valid_chunks)

        if not ensemble:
            # Return ensemble mean
            _forecast = _forecast.mean(dim="number")
        # Build "month" dimension (redundant because forecast only includes 1 year)    
        _forecast = _forecast.groupby("time.month").mean()
        # Compute Impact (Forecast Effect - Baseline Effect)
        impact = _forecast - _baseline
        # Save attributes
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
    #Compute the global impact region by summing across spatial "group_dim"
    if rate:
        #Pop-weight each region
        pop = socioeconomics["population"].sel(region=impact.region)
        return (impact * pop).sum(dim=group_dim) / pop.sum(dim=group_dim)
    return impact.sum(dim=group_dim)

### Analysis Functions ###
def xarray_to_gpd(data, polygons, crs="ESRI:54030"):
    # Merge xarray with dim "region" with Impact Region Geopandas
    _polygons_data = polygons.merge(
        data.to_dataframe(name=data.name or "value").reset_index(),
        on="region",
    )
    # Clip Antarctica
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
def redistribute_adm1(impact, config, chunk_size=4):
    value_name = impact.name or "value"
    # List of impact regions
    col_regions = {(h,) for h in impact["region"].values}

    # Total Deaths: "per_source", "kind = extensive"
    # Death Rates: "per_destination", "kind = intensive"
    kind = "intensive" if config.rate else "extensive"
    # Get weights from pre-defined files
    weights = cilreg.fetch_weights(
        "gadm41-adm1-per-destination" if config.rate else "gadm41-adm1-per-source"
    )
    # store chunked results
    results = []
    # Check for ensemble members
    numbers = impact["number"].values if "number" in impact.dims else [None]
    # Process for each ensemble member chunk to manage memory (Other option could be dask-pandas)
    for start in range(0, len(numbers), chunk_size):
        chunk = (
            impact.isel(number=slice(start, start + chunk_size))
            if "number" in impact.dims else impact
        )
        #xarray to dataframe
        df_chunk = xarray_to_gpd(chunk, config.polygons)
        # Align region names with weights file
        df_chunk = df_chunk.rename(columns={"region": "hierid", value_name: "value"})
        # keep only region, value, and dim columns
        keep_cols = ["hierid", "value"] + list(config.dims)
        if "month" in df_chunk.columns:
            keep_cols.append("month")
        df_chunk = df_chunk[keep_cols]
        # Apply weights from cilreg
        out = cilreg.apply_weights(
            weights, # pre-computed from input file
            df_chunk, 
            kind=kind, 
            weight="pop", 
            value_col="value",
            data_version="world-combo-201710",
            restrict_to_sources=col_regions, 
            allow_partial_coverage=True,
            policies=SourceUnitPolicies(
                on_unmatched="skip", 
                on_zero_weight="skip", # handle zero pop
                on_absent_from_data="skip" #handel Antarctica ISO
            ),
        ).frame
        results.append(out)
        del df_chunk, out  # free memory before next iteration

    adm1 = pd.concat(results, ignore_index=True)
    adm1["GID_1"] = adm1["GID_1"].astype(str)
    # Manage National-level ADM1 (Typically missing an GID_1, fill with GID_0)
    mask = adm1["GID_1"] == ""
    adm1.loc[mask, "GID_1"] = adm1.loc[mask, "GID_0"]

    index_cols = ["GID_1"] + list(config.dims)
    if "month" in adm1.columns:
        index_cols.append("month")
    # Add back GID_0 lookup for alignment (if it gets left in the dataframe you get GID_0xGID_1)
    gid0_lookup = adm1.drop_duplicates("GID_1").set_index("GID_1")["GID_0"]

    # Transform dataframe back to xarray
    da = adm1.set_index(index_cols)["value"].to_xarray()
    da = da.assign_coords(GID_0=("GID_1", gid0_lookup.reindex(da["GID_1"].values).values))

    return da

def aggregate_by_iso(ds, polygon, operation="sum"):
    # Add ISO (from shapefile gpd) to xarray
    # align ISO and Impact region dataframes
    iso_map = polygon["ISO"].reindex(ds["region"].values)
    ds = ds.assign_coords(ISO=("region", iso_map.to_numpy(dtype=object)))

    grouped = ds.groupby("ISO")
    if operation == "sum":
        # Sum total deaths
        ds = grouped.sum(dim="region")
    else:
        raise ValueError(f"Unsupported operation: {operation!r}")
    # Dissolve Impact Region geometry to ISO level
    polygon = polygon.copy()
    polygon["geometry"] = polygon["geometry"].buffer(0)
    polygon = polygon.dissolve(by="ISO", aggfunc="first")

    return ds, polygon

def aggregate_impact(impact, config, group_level): #Needs validation
    if group_level == "IR":
        # Add ISO (from shapefile gpd) to xarray to match data across spatial resolutions
        # align ISO and Impact region dataframes
        iso_map = config.polygons["ISO"].reindex(impact["region"].values)
        impact = impact.assign_coords(ISO=("region", iso_map.to_numpy(dtype=object)))
        return impact, "region", ["region", "ISO"]

    if group_level == "ISO":
        if config.rate:
            # Pop-weighting total deaths required
            raise ValueError("ISO grouping is not supported when rate=True")
        impact, _ = aggregate_by_iso(impact, config.polygons, operation="sum")
        return impact, "ISO", ["ISO"]

    if group_level == "ADM1":
        impact = redistribute_adm1(impact, config)
        return impact, "GID_1", ["GID_0", "GID_1"]

    raise ValueError(f"Unsupported group_level: {group_level!r}")

###Output Functions ###
def _baseline_tag(baseline_period):
    start_year = baseline_period.start[:4]
    stop_year = baseline_period.stop[:4]
    return f"{start_year}-{stop_year}"

def dataset_to_dataframe(ds):
    if len(ds.dims) == 0:
        return pd.DataFrame({k: [v.values.item()] for k, v in ds.data_vars.items()})
    return ds.to_dataframe().reset_index()

def make_csv(
    effect,
    config: ImpactConfig,
    group_level="IR",
    filename_template="{version}_{hotonly}_{scope}_{rate_l}_{stat_scope}_{group_level}_{baseline}.csv",
    output_scope=["regional_monthly", "regional_6mo", "global_monthly", "global_6mo"],
):
    rate_l = "rate" if config.rate else "total"
    baseline_tag = _baseline_tag(config.baseline_period)

    impact = config.compute_impact(effect.chunk({dim: -1 for dim in config.dims}), ensemble=True)
    impact = impact.sel(month=config.months)

    polygon = config.polygons
    # Aggregate Impact Regions to group_level
    impact, merge_key, base_cols = aggregate_impact(impact, config, group_level)

    stat_cols = ["median", "p17", "p83", "likely_range_IPCC", "mean", "std", "min", "max", "p10", "p90"]
    # Step through outputs listed in 'output_scope"
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
                scope="monthly",
                stat_scope="",
                group_level=group_level,
                baseline=baseline_tag,
            ),
            index=False,
        )

    if "regional_6mo" in output_scope:
        mo6 = impact.sum(dim="month")
        _polygons_mo6 = dataset_to_dataframe(compute_stats(mo6, dim=config.dims))
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
                baseline=baseline_tag,
            ),
            index=False,
        )

    if "global_monthly" in output_scope or "global_6mo" in output_scope:
        # Compute global total before getting stats
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
                    baseline=baseline_tag,
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
                baseline=baseline_tag,
            ),
                index=False,
            )



##### Land Only #####
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

