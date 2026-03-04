# VT-algae-blooms
CS 5540 class project to use machine learning techniques to predict algae blooms in Vermont

## Setup
```bash
# install uv
pip install uv

# Activate the virtual environment
# Note that you must be at the project root directory
source .venv/bin/activate

# Install project packages into the virtual environment
uv sync
```


### Team

Charlie Davidson  
Joshua Fishbein  
Helen Flannery  
Alex Schaefer
 
### Project
 
We will develop a machine learning model that will predict algae blooms in selected bodies of water in Vermont. The data set we will be using comes from the [Vermont Cyanobacteria Tracker](https://www.healthvermont.gov/environment/tracking/cyanobacteria-blue-green-algae-tracker) from the Vermont Department of Health, and we will combine that data with weather information from the National Weather Service. Potential research questions include:
 
- Is it possible to predict when a bloom will happen?
- Is it possible to predict characteristics of algae blooms (e.g. their size, duration, toxicity)?
 
### Tasks
 
- Acquire the data
  - Algae blooms
  - Weather
- Explore the data
  - Create test & cross-validation sets
  - Create categorical attributes and quantitative variables
- Prepare for machine learning algorithms
  - Clean the data
  - Prepare text and categorical attributes (e.g. one-hot encoding if necessary)
  - Feature scaling & transformation (e.g. standardize maximums & minimums; normalize attributes)
- Select & train on model(s)
  - Test & evaluate on training set (we will each try a different model)
  - Do cross-validation
  - Feature selection
- Fine-tune the model(s)
  - Evaluate on test set
- Write project report & presentation

### Data

The target data comes from the VT Department of Health [Cyanobacteria (Blue-Green Algae) Tracker](https://anrweb.vermont.gov/vct/vct/LongTermMonitoringLakes.aspx)

The features used in our model come from the following data sources

- The VT Department of Health [Cyanobacteria (Blue-Green Algae) Tracker](https://anrweb.vermont.gov/vct/vct/LongTermMonitoringLakes.aspx). This data includes features for cyanobacteria taxa observed at different sites

- The Department of Environmental Conservation at the Vermont Agency of Natural Resources (DEC). This data set includes features for levels of different elements, many of which signal the presence of agricultural or municipal runoff (nitrogen, phosphorus, carbon, oxygen); pH; cyanobacteria quantities; water temperature; and water opacity. (https://anrweb.vermont.gov/dec/_dec/LongTermMonitoringLakes.aspx?_gl=1*1h6x1l7*_ga*MTc2NDE4MDg1Ny4xNzY4MTg0Nzc2*_ga_V9WQH77KLW*czE3Njk5MTU3MTMkbzQkZzEkdDE3Njk5MTYwMzkkajYwJGwwJGgw)

- The U.S. Geological Service National Water Information System (USGS NWIS). This data set includes features for water temperature, water surface level, and site depth. (https://www.usgs.gov/tools/national-water-information-system-nwis-mapper)

- The NOAA National Centers for Environmental Information (NOAA NCEI). This data set includes features for air temperature, cloud cover, daylight hours (sunrise/sunset to align with cloud cover), precipitation, wind speed, and wind direction. We will be using the Burlington (BTV) airport station code (USW00014742). (https://www.ncei.noaa.gov/cdo-web/)

During exploratory data analysis, the following sources were used:

- We used the Vermont Department of Environmental Conservation's Long Term Monitoring Lakes Project documentation to map DEC monitoring stations to the VCT algae blooms data. That data can be found at https://anrweb.vermont.gov/DEC/_DEC/LongTermMonitoringLakes.aspx. The formula we used to convert the decimal minutes latitude/longitude format in the DEC's data to the decimal degrees format in the VCT data is below.

- We used Lake Champlain shapefiles from [vermont.gov](https://geodata.vermont.gov/datasets/vt-lake-champlain-extracted-from-vhdcarto-polygon/about) to visualize monitoring site locations to map the VT Deparment of Health lake monitoring stations to the most relevant VT Department of Environmental Conservation lake monitoring stations

### Methods

1. Data was downloaded from the links in the Data section, saved as zipfiles and loaded into the
   `data/raw_files directory`
2. The raw files were aggregated into unified csvs using the following:
```shell
notebooks/vtab_combine_vct_data_files.jpynb

notebooks/vtab_usgs_nwis_data_pull.jpynb

python scripts/combine_dec_yearly_reports.py \
    -i data/raw_files/vt_dec_yearly_reports.zip \
    -o data/unified_csvs/vt_dec.csv

python scripts/combine_vct_yearly_reports.py \
    -i data/raw_files/vct_yearly_reports.zip \
    -o data/unified_csvs/vct.csv
```
3. The unified csv files were converted into usable features using the following:
```shell
notebooks/vtab_vct_prep_features_and_targets.jpynb

notebooks/vtab_usgs_prep_features_and_targets.jpynb

python scripts/create_dec_features.py \
    -o data/unified_csvs/vt_dec_unified_prepped.csv
```
4. The unified csv files were joined into a single merged feature/target file using the following:
```shell
notebooks/vtab_big_data_join.jpynb
```
