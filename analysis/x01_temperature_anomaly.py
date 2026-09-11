#Temperature anomaly ensemble mean

import os
from dotenv import load_dotenv
import xarray as xr

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import date

import analysis_utils
import isku_utils

import importlib

importlib.reload(analysis_utils)
importlib.reload(isku_utils)

load_dotenv()
DATA_DIR = os.environ["DATA_DIR"]
IMPACT_REGION_POLYGONS = os.environ["POREALLAS_REGIONS_POLYGONS_URI"]
SOCIOECONOMICS_URI = os.environ["POREALLAS_SOCIOECONOMICS_URI"]

# Impact Regions
_polygons = (
    gpd.read_parquet(os.path.join(DATA_DIR, IMPACT_REGION_POLYGONS))
    .rename(columns={"hierid": "region"})
    .set_index("region")
    .set_crs(epsg=4326)  # Assuming the data is WGS-82.
)

# Socioeconomics
socioeconomics = xr.open_zarr(os.path.join(DATA_DIR, SOCIOECONOMICS_URI))
socioeconomics = socioeconomics.sel(year=2026)[
    ["pop", "gdppc", "iso3"]
]

# Baseline period Monthly Climatology
BASELINE_PERIOD = slice("1995-01-01", "2014-12-31")
# Define Forecast Months
FC_MONTHS = [9, 10, 11, 12, 1, 2]
FC_PERIOD = slice("2026-09-01", "2027-02-28")
# Define Forecast
POREALLAS_PARSED_FORECAST_URI = "/home/emily_zuetell/projects/poreallas/data/parsed/v20260909_parsed_forecast.zarr"

forecast = xr.open_zarr(
        POREALLAS_PARSED_FORECAST_URI)

# Compute Anomaly
forecast_base = forecast.sel(time=BASELINE_PERIOD).groupby('time.month').mean().mean(dim = 'number')
### Anomaly
# Analysis Period (2026-2027)
forecast_analysis = forecast.sel(time=FC_PERIOD).resample(time='MS').mean()

# Subtract Monthly Climatology
forecast_anomaly = forecast_analysis.groupby('time.month') - forecast_base

forecast_ir_anomaly = isku_utils.grid_to_ir(forecast_anomaly['tas'].mean(dim = 'number'))
forecast_ir_anomaly = forecast_ir_anomaly.rename('anomaly')

df = forecast_ir_anomaly.drop_vars('month').to_dataframe().reset_index()
df['time'] = df['time'].apply(lambda t: t.strftime('%Y-%m-%d'))
df = df.pivot(index='region', columns='time', values='anomaly')
df['6-Mo Mean'] = df.mean(axis=1)

filename = f'v{date.today().strftime("%Y%m%d")}_forecast_anomaly.csv'
df.to_csv(filename)
# write metadata:
with open(filename, 'w') as f:
    f.write(f'# generated: {date.today().isoformat()}\n')
    f.write(f'# source: {POREALLAS_PARSED_FORECAST_URI}\n')
    f.write(f'# baseline_period: {BASELINE_PERIOD.start} to {BASELINE_PERIOD.stop}\n')
    f.write(f'# forecast_period: {FC_PERIOD.start} to {FC_PERIOD.stop}\n')
    df.to_csv(f)