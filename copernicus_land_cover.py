import os
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from sentinelhub import (
    SHConfig,
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    bbox_to_dimensions,
)

# ============================================================
# USER SETTINGS
# ============================================================

CATCHMENT_PATH = r"/home/rpb/majiconsult/trifinio/boundaries/trifinio_hydro_atlas_lvl10.shp"
OUT_DIR = Path("/home/rpb/majiconsult/trifinio/copernicus_landcover_100m")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATCHMENT_ID_FIELD = "HYBAS_ID"
TARGET_CRS = "EPSG:4326"
YEARS = [2015, 2016, 2017, 2018, 2019]

# If True, do not download again when raster already exists locally
SKIP_IF_DOWNLOADED = True

# Load copernicus account - requires a .env file with copernicus account seetings
load_dotenv()
SH_CLIENT_ID = os.getenv("SH_CLIENT_ID")
SH_CLIENT_SECRET = os.getenv("SH_CLIENT_SECRET")

if not SH_CLIENT_ID or not SH_CLIENT_SECRET:
    raise RuntimeError("Please set SH_CLIENT_ID and SH_CLIENT_SECRET in your environment.")

COLLECTION_ID = "35fecfec-8a73-4723-bb08-b775f283a535"
BAND_NAME = "Discrete_Classification"
RESOLUTION_M = 100

# ============================================================
# CLASS LEGENDS AND GROUPING
# ============================================================

def make_class_legend() -> pd.DataFrame:
    legend = [
        (0, "Unknown / No data"),
        (20, "Shrubs"),
        (30, "Herbaceous vegetation"),
        (40, "Cultivated and managed vegetation / agriculture"),
        (50, "Urban / built-up"),
        (60, "Bare / sparse vegetation"),
        (70, "Snow and ice"),
        (80, "Permanent water bodies"),
        (90, "Herbaceous wetland"),
        (100, "Moss and lichen"),
        (111, "Closed forest, evergreen needleleaf"),
        (112, "Closed forest, evergreen broadleaf"),
        (113, "Closed forest, deciduous needleleaf"),
        (114, "Closed forest, deciduous broadleaf"),
        (115, "Closed forest, mixed"),
        (116, "Closed forest, unknown"),
        (121, "Open forest, evergreen needleleaf"),
        (122, "Open forest, evergreen broadleaf"),
        (123, "Open forest, deciduous needleleaf"),
        (124, "Open forest, deciduous broadleaf"),
        (125, "Open forest, mixed"),
        (126, "Open forest, unknown"),
        (200, "Oceans / seas"),
    ]
    return pd.DataFrame(legend, columns=["class_code", "class_name"])


# Grouping requested:
# - all forest classes together
# - moss/lichen + herbaceous vegetation + sparse vegetation together
CLASS_TO_GROUP = {
    0: "Unknown / No data",
    20: "Herbaceous-sparse-shrubs-moss",
    30: "Herbaceous-sparse-shrubs-moss",
    40: "Agriculture",
    50: "Urban / built-up",
    60: "Herbaceous-sparse-shrubs-moss",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous-sparse-shrubs-moss",
    100: "Herbaceous-sparse-shrubs-moss",
    111: "Closed Forest",
    112: "Closed Forest",
    113: "Closed Forest",
    114: "Closed Forest",
    115: "Closed Forest",
    116: "Closed Forest",
    121: "Open Forest",
    122: "Open Forest",
    123: "Open Forest",
    124: "Open Forest",
    125: "Open Forest",
    126: "Open Forest",
    200: "Oceans / seas",
}


def make_grouped_legend() -> pd.DataFrame:
    grouped = [
        ("Unknown / No data", "No data or unmapped"),
        ("Forest", "All open and closed forest classes"),
        ("Herbaceous-sparse-shrubs-moss", "Grouped herbaceous vegetation, herbaceous wetland, sparse vegetation, shrubs, moss and lichen"),
        ("Agriculture", "Cultivated and managed vegetation / agriculture"),
        ("Urban / built-up", "Urban and built-up areas"),
        ("Snow and ice", "Snow and ice"),
        ("Permanent water bodies", "Inland water bodies"),
        ("Oceans / seas", "Marine water"),
    ]
    return pd.DataFrame(grouped, columns=["group_class", "group_description"])


# ============================================================
# HELPERS
# ============================================================

def build_config() -> SHConfig:
    cfg = SHConfig()
    cfg.sh_client_id = SH_CLIENT_ID
    cfg.sh_client_secret = SH_CLIENT_SECRET
    cfg.sh_base_url = "https://sh.dataspace.copernicus.eu"
    cfg.sh_token_url = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    )
    return cfg


def build_evalscript() -> str:
    return f"""
//VERSION=3
function setup() {{
  return {{
    input: [{{
      bands: ["{BAND_NAME}", "dataMask"]
    }}],
    output: {{
      bands: 2,
      sampleType: "UINT16"
    }}
  }};
}}

function evaluatePixel(sample) {{
  return [sample.{BAND_NAME}, sample.dataMask];
}}
"""


def download_landcover_year(catchments: gpd.GeoDataFrame, year: int, config: SHConfig) -> Path:
    out_path = OUT_DIR / f"copernicus_lc_100m_{year}.tif"

    if SKIP_IF_DOWNLOADED and out_path.exists():
        print(f"Using existing raster for {year}: {out_path}")
        return out_path

    minx, miny, maxx, maxy = catchments.total_bounds
    bbox = BBox((minx, miny, maxx, maxy), crs=CRS.WGS84)

    byoc = DataCollection.define_byoc(collection_id=COLLECTION_ID)
    size = bbox_to_dimensions(bbox, resolution=RESOLUTION_M)

    request = SentinelHubRequest(
        evalscript=build_evalscript(),
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=byoc,
                time_interval=(
                    f"{year}-01-01T00:00:00Z",
                    f"{year}-12-31T23:59:59Z",
                ),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )

    data = request.get_data()[0]
    class_arr = data[:, :, 0].astype(np.uint16)
    mask_arr = data[:, :, 1].astype(np.uint16)

    transform = from_bounds(minx, miny, maxx, maxy, class_arr.shape[1], class_arr.shape[0])

    profile = {
        "driver": "GTiff",
        "height": class_arr.shape[0],
        "width": class_arr.shape[1],
        "count": 2,
        "dtype": rasterio.uint16,
        "crs": TARGET_CRS,
        "transform": transform,
        "compress": "lzw",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(class_arr, 1)
        dst.write(mask_arr, 2)
        dst.set_band_description(1, "landcover_class")
        dst.set_band_description(2, "dataMask")

    print(f"Downloaded raster for {year}: {out_path}")
    return out_path


def zonal_class_stats(
    catchments: gpd.GeoDataFrame,
    raster_path: Path,
    year: int,
    catchment_id_field: str,
) -> pd.DataFrame:
    rows = []

    with rasterio.open(raster_path) as src:
        class_band = src.read(1)
        data_mask = src.read(2)
        transform = src.transform

        valid = data_mask == 1
        pixel_area_ha = (RESOLUTION_M * RESOLUTION_M) / 10000.0

        for _, feat in catchments.iterrows():
            catchment_id = feat[catchment_id_field]
            geom = [feat.geometry]

            mask = geometry_mask(
                geometries=geom,
                transform=transform,
                invert=True,
                out_shape=class_band.shape,
                all_touched=False,
            )

            values = class_band[mask & valid]

            if values.size == 0:
                continue

            classes, counts = np.unique(values, return_counts=True)
            total_valid_pixels = counts.sum()
            total_valid_area_ha = total_valid_pixels * pixel_area_ha

            for class_code, pixel_count in zip(classes, counts):
                area_ha = pixel_count * pixel_area_ha
                area_pct = (area_ha / total_valid_area_ha * 100) if total_valid_area_ha > 0 else 0.0

                rows.append(
                    {
                        "catchment": catchment_id,
                        "year": year,
                        "class_code": int(class_code),
                        "pixel_count": int(pixel_count),
                        "area_ha": float(area_ha),
                        "area_pct": float(area_pct),
                        "total_valid_area_ha": float(total_valid_area_ha),
                    }
                )

    return pd.DataFrame(rows)


def build_grouped_stats(detail_df: pd.DataFrame) -> pd.DataFrame:
    grouped_df = detail_df.copy()
    grouped_df["group_class"] = grouped_df["class_code"].map(CLASS_TO_GROUP).fillna("Other")

    grouped_df = (
        grouped_df.groupby(["catchment", "year", "group_class"], as_index=False)
        .agg(
            pixel_count=("pixel_count", "sum"),
            area_ha=("area_ha", "sum"),
            total_valid_area_ha=("total_valid_area_ha", "first"),
        )
    )

    grouped_df["area_pct"] = np.where(
        grouped_df["total_valid_area_ha"] > 0,
        grouped_df["area_ha"] / grouped_df["total_valid_area_ha"] * 100,
        0.0,
    )

    grouped_df = grouped_df[
        ["catchment", "year", "group_class", "pixel_count", "area_ha", "area_pct", "total_valid_area_ha"]
    ].sort_values(["catchment", "year", "group_class"]).reset_index(drop=True)

    return grouped_df


# ============================================================
# MAIN
# ============================================================

def main():
    config = build_config()
    catchments = gpd.read_file(CATCHMENT_PATH).to_crs(TARGET_CRS)

    all_stats = []

    for year in YEARS:
        print(f"Processing {year}...")
        raster_path = download_landcover_year(catchments, year, config)

        df = zonal_class_stats(
            catchments=catchments,
            raster_path=raster_path,
            year=year,
            catchment_id_field=CATCHMENT_ID_FIELD,
        )
        all_stats.append(df)

    final_df = pd.concat(all_stats, ignore_index=True)
    final_df = final_df.sort_values(["catchment", "year", "class_code"]).reset_index(drop=True)

    # rounding
    for col in ["area_ha", "area_pct", "total_valid_area_ha"]:
        final_df[col] = final_df[col].round(2)

    # detailed outputs
    detail_csv = OUT_DIR / "copernicus_landcover_100m_stats_by_catchment.csv"
    detail_legend_csv = OUT_DIR / "copernicus_landcover_100m_class_legend.csv"

    final_df.to_csv(detail_csv, index=False)
    make_class_legend().to_csv(detail_legend_csv, index=False)

    # grouped outputs
    grouped_df = build_grouped_stats(final_df)
    for col in ["area_ha", "area_pct", "total_valid_area_ha"]:
        grouped_df[col] = grouped_df[col].round(2)

    grouped_csv = OUT_DIR / "copernicus_landcover_100m_stats_grouped_by_catchment.csv"
    grouped_legend_csv = OUT_DIR / "copernicus_landcover_100m_grouped_legend.csv"

    grouped_df.to_csv(grouped_csv, index=False)
    make_grouped_legend().to_csv(grouped_legend_csv, index=False)

    print(f"Detailed stats: {detail_csv}")
    print(f"Detailed legend: {detail_legend_csv}")
    print(f"Grouped stats: {grouped_csv}")
    print(f"Grouped legend: {grouped_legend_csv}")


if __name__ == "__main__":
    main()