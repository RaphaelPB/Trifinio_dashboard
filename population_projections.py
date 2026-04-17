import os

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats

#%% Catchments
catchment_path = r"/home/rpb/majiconsult/trifinio/boundaries/trifinio_hydro_atlas_lvl10.shp"
catch_id_field = "HYBAS_ID"
bCRS = 'EPSG:4326'

#%% Population
#source: https://doi.org/10.7927/q7z9-9r69
#download: https://sedac.ciesin.columbia.edu/data/set/popdynamics-1-km-downscaled-pop-base-year-projection-ssp-2000-2100-rev01
pop_data =r'/home/rpb/majiconsult/trifinio/socioeconomic/GlobalPopulationProjections'
years = [2020,2030,2040,2050]
SSPs = ['ssp1','ssp3','ssp5']

#%% Water use (liters/day/capita)
per_capita_use_lpd = {
    "El Salvador": {"Total": 64,},
    "Guatemala":   {"Total": 48,},
    "Honduras":    {"Total": 31,},
}

#%% Export path
export_path = r"/home/rpb/majiconsult/trifinio/socioeconomic/ssp_population_by_catchment.csv"


#%% Process data
#%% Load catchments
catch_shape = gpd.read_file(catchment_path).to_crs(bCRS)

#%% Store results here
results = []

#%% Loop
for scen in SSPs:
    for year in years:
        for type in ['Urban', 'Rural', 'Total']:
            print(f"Processing {scen} {year} {type}")

            # adapts data path to type
            prefix = '_'
            if type == 'Urban':
                prefix = 'urb'
            elif type == 'Rural':
                prefix = 'rur'
            data = os.path.join(scen, f"{type}", 'ASCII', f"{scen}{prefix}{year}.txt")
            raster = os.path.join(pop_data, data)

            stats = zonal_stats(
                vectors=catch_shape,
                raster=raster,
                all_touched=False,
                stats='sum'
            )

            df = pd.DataFrame(stats)

            # Add identifiers
            df["type"] = type
            df["catchment"] = catch_shape[catch_id_field].values
            df["SSP"] = scen
            df["year"] = year
            df["country"] = catch_shape['CNTR_NAME']

            # Rename value column
            df = df.rename(columns={"sum": "population"})

            # Compute annual water consumption (m3/year)
            df["water_consumption_Mm3_year"] = df.apply(
                lambda row: (
                        row["population"]
                        * per_capita_use_lpd[row["country"]][row["type"]]
                        * 365/ 10**9  #l/day to Mm3/year
                    if row["country"] in per_capita_use_lpd.keys() and row["type"] in per_capita_use_lpd[row["country"]].keys()
                    else np.nan
                ),
                axis=1,
            )

            results.append(df)


#%% Combine all results
final_df = pd.concat(results, ignore_index=True)

#%% round
final_df["population"] = final_df["population"].round(0)
final_df["water_consumption_Mm3_year"] = final_df["water_consumption_Mm3_year"].round(3)


#%% Reorder columns
final_df = final_df[["catchment", "SSP", "year", "type", "population", "water_consumption_Mm3_year"]]

#%% Save
final_df.to_csv(export_path, index=False)