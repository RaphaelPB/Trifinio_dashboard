import geopandas as gpd

## MAKE CSV TABLE FROM HYDRO ATLAS

# === 1. Load your shapefile ===
shapefile_path = r"/home/rpb/majiconsult/trifinio/boundaries/trifinio_hydro_atlas_lvl10.shp"
gdf = gpd.read_file(shapefile_path)

# === 2. Inspect columns (optional but recommended) ===
print(gdf.columns)

# === 3. Drop geometry (Power BI doesn't need it for table view) ===
df = gdf.drop(columns="geometry")

# === 4. Save to CSV ===
output_csv = r"/home/rpb/majiconsult/trifinio/boundaries/trifinio_hydro_atlas_lvl10.csv"
df.to_csv(output_csv, index=False)

print("CSV successfully exported!")