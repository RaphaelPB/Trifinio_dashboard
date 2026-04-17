import geopandas as gpd
import pandas as pd

#Define country of a catchment (when it does not fit exactly)
def country_of_catch(shape1,shape2,id1,id2):
    intersect = shape1.overlay(shape2,how='intersection')
    intersect['area'] = intersect.to_crs({'proj':'cea'}).area/10**6 #area in km2
    table = intersect[[id1,id2,'area']]
    matrix=pd.DataFrame(index=shape1[id1],columns=shape2[id2],dtype=float)
    for wc in shape1[id1].values:
        a=table.loc[table[id1]==wc].set_index(id2)
        for cc in a.index:
            matrix.loc[wc,cc]=a['area'].loc[cc]
    conv=matrix.idxmax(axis=1)
    shape1[id2]=shape1[id1].apply(lambda k: conv.loc[k])


## User input

## MAKE CSV TABLE FROM HYDRO ATLAS

# hydroatlas
hydroatlas_path = r"/home/rpb/majiconsult/trifinio/boundaries/trifinio_hydro_atlas_lvl10.shp"
basin_id = 'HYBAS_ID'
# countries
country_path = "/home/rpb/majiconsult/trifinio/boundaries/CNTR_RG_20M_2024_4326/CNTR_RG_20M_2024_4326.shp"
country_id = "CNTR_NAME"
# rename basins (Hydroatlas)
rename_main_basin = True
main_basin_name = {7100002700:'Lempa', 7100053240:'Motagua', 7100053320:'Higuito'}
#Projection crs - IT is VERY important to ensure that all data have the same projection when performing calculations
bCRS='EPSG:4326'

# === 1. Load hydroatlas shapefile ===

gdf = gpd.read_file(hydroatlas_path)

# === 2. Inspect columns (optional but recommended) ===
print(gdf.columns)

# == Rename basins ==
if rename_main_basin:
    gdf['basin_name']=gdf['MAIN_BAS'].map(lambda x: main_basin_name[x] if x in main_basin_name.keys() else '')

# == Get country of catchment ==
country_shape=gpd.read_file(country_path).to_crs(bCRS)
if country_id not in gdf.columns:
    country_of_catch(gdf, country_shape, basin_id, country_id)

# == Fix invalid geometries ==
gdf = gdf[gdf.geometry.notna()].copy()
gdf["geometry"] = gdf.geometry.make_valid()
# Optional: dissolve multipart artifacts back by basin id
gdf = gdf.dissolve(by=basin_id, as_index=False)

# Project to metric CRS
gdf_m = gdf.to_crs(3857)   # meters
# shrink polygons inward by 5 meters
gdf_m["geometry"] = gdf_m.buffer(-25)
# back to WGS84
gdf_m = gdf_m.to_crs(4326)

# == SAve hydrotlas ==
gdf.to_file(hydroatlas_path)

# === 3. Drop geometry (Power BI doesn't need it for table view) ===
df = gdf.drop(columns="geometry")

# === 4. Save to CSV ===
output_csv = r"/home/rpb/majiconsult/trifinio/boundaries/trifinio_hydro_atlas_lvl10.csv"
df.to_csv(output_csv, index=False)

print("CSV successfully exported!")