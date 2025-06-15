# This algorithm extracts reflectance values from a raster image at the locations specified in a CSV file.

# INPUTS
# csv_path: Path to the CSV file containing sample points and their reflectance values.
# raster_path: Path to the raster image from which reflectance values will be extracted.

# OUTPUTS
# A DataFrame containing the original data along with the extracted reflectance values for each band.

#%%------------------------------------------------------------------------------
# IMPORTS
#--------------------------------------------------------------------------------
from pathlib import Path
import rasterio
from shapely.geometry import Point
import numpy as np
import pandas as pd
import geopandas as gpd
import ast

#%%------------------------------------------------------------------------------
# INPUTS
#--------------------------------------------------------------------------------
csv_path    = Path(r"H:\Meu Drive\Artigo_Mestrado\03_RSS\RSS_candidateSamples_filtered_20250612.csv")
raster_path = Path(r"H:\Meu Drive\Artigo_Mestrado\01_Imagens\Sentinel2\S2_resampled60m.tif")

#%%------------------------------------------------------------------------------
# Load files
#--------------------------------------------------------------------------------
df = pd.read_csv(csv_path)

src = rasterio.open(raster_path)
#%%------------------------------------------------------------------------------
# Get coordinate values
#--------------------------------------------------------------------------------
df['points'] = df['centroid'].apply(lambda s: Point(*ast.literal_eval(s)))
# Convert points to a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry='points', crs=src.crs)
coord_list = [(x, y) for x, y in zip(gdf["points"].x, gdf["points"].y)]
#%%------------------------------------------------------------------------------
# Extrac
#--------------------------------------------------------------------------------
gdf["value"] = [x for x in src.sample(coord_list)]

bands =["blue", "green", "red", "nir"]
for i, band in enumerate(bands):
    gdf[band] = [x[i] for x in gdf["value"]]

gdf.drop(columns=["centroid", "value"], inplace=True)
gdf['ndvi'] = (gdf['nir'] - gdf['red']) / (gdf['nir'] + gdf['red'])

gdf[bands] = gdf[bands].astype(float) / 10000.0  # Convert to reflectance values

gdf.to_csv(  Path(csv_path.parent, csv_path.stem + "_reflectances.csv"), index=False)