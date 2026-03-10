import os

DATA_DIR = "data"

RAW_FILES_DIR = os.path.join(DATA_DIR, "raw_files")
VCT_RAW_PATH = os.path.join(RAW_FILES_DIR, "vct_env-epht-cyanobacteria-season-summaries.zip")
DEC_RAW_PATH = os.path.join(RAW_FILES_DIR, "vt_dec_yearly_reports.zip")
NOAA_RAW_PATH = os.path.join(RAW_FILES_DIR, "NOAA_weather_data.csv")
USGS_RAW_PATH = os.path.join(RAW_FILES_DIR, "USGS_NWIS_data_2026_02_09.csv")

UNIFIED_CSVS_DIR = os.path.join(DATA_DIR, "unified_csvs")
VCT_UNIFIED_PATH = os.path.join(UNIFIED_CSVS_DIR, "vct.csv")
DEC_UNIFIED_PATH = os.path.join(UNIFIED_CSVS_DIR, "dec.csv")
NOAA_UNIFIED_PATH = os.path.join(UNIFIED_CSVS_DIR, "noaa.csv")
USGS_UNIFIED_PATH = os.path.join(UNIFIED_CSVS_DIR, "usgs.csv")

FEATURES_AND_TARGETS_DIR = os.path.join(DATA_DIR, "features_and_targets")
TARGETS_PATH = os.path.join(FEATURES_AND_TARGETS_DIR, "targets.csv")
VCT_FEATURES_PATH = os.path.join(FEATURES_AND_TARGETS_DIR, "vct_features.csv")
DEC_FEATURES_PATH = os.path.join(FEATURES_AND_TARGETS_DIR, "dec_features.csv")
NOAA_FEATURES_PATH = os.path.join(FEATURES_AND_TARGETS_DIR, "noaa_features.csv")
USGS_FEATURES_PATH = os.path.join(FEATURES_AND_TARGETS_DIR, "usgs_features.csv")

VCT_TO_DEC_SITE_MAPPING_PATH = os.path.join(DATA_DIR, "data_dictionaries", "VCT_to_DEC_site_mappings.json")
