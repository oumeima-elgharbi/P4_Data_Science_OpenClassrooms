# P4_Data_Science_OpenClassrooms

### Interesting variables
Source : https://data.seattle.gov/dataset/2016-Building-Energy-Benchmarking/2bpz-gwpy

### Unique ID, Primary Key

- OSEBuildingID : A unique identifier assigned to each property covered by the Seattle Benchmarking Ordinance for tracking and identification purposes.


### Categorical variables (qualitative)

#### Building characteristics :

- BuildingType : City of Seattle building type classification.

- PrimaryPropertyType : The primary use of a property (e.g. office, retail store). Primary use is defined as a function that accounts for more than 50% of a property. This is the Property Type - EPA Calculated field from Portfolio Manager.
  - EPA = Environmental Protection Agency (United States)

- **ListOfAllPropertyUseTypes** : All property uses reported in Portfolio Manager

- **LargestPropertyUseType** : The largest use of a property (e.g. office, retail store) by GFA.

- **SecondLargestPropertyUseType** : The second largest use of a property (e.g. office, retail store) by GFA.

- **ThirdLargestPropertyUseType** : The third largest use of a property (e.g. office, retail store) by GFA.


### Numeric variables (quantitative) :

#### Localisation :

- Latitude : Property latitude.

- Longitude : Property longitude.

- ZipCode : Property zip

- CouncilDistrictCode : Property City of Seattle council district.


#### Building characteristics :

- DataYear : Calendar year (January-December) represented by each data record.

- YearBuilt : Year in which a property was constructed or underwent a complete renovation.

- NumberofBuildings : Number of buildings included in the property's report. In cases where a property is reporting as a campus, multiple buildings may be included in one report.

- NumberofFloors : Number of floors reported in Portfolio Manager

- **PropertyGFATotal** : Total building and parking gross floor area.

- **PropertyGFAParking** : Total space in square feet of all types of parking (Fully Enclosed, Partially Enclosed, and Open).

- **PropertyGFABuilding(s)** : Total floor space in square feet between the outside surfaces of a building’s enclosing walls. This includes all areas inside the building(s), such as tenant space, common areas, stairwells, basements, storage, etc.

- **LargestPropertyUseTypeGFA** : The gross floor area (GFA) of the largest use of the property.

- **SecondLargestPropertyUseTypeGFA** : The gross floor area (GFA) of the second largest use of the property.

- **ThirdLargestPropertyUseTypeGFA** : The gross floor area (GFA) of the third largest use of the property.

  - N.B. : Gross floor area (GFA) = the total floor area contained within the building measured to the external face of the external walls. (Wikipedia)


#### Energy : 

- ENERGYSTARScore : An EPA calculated 1-100 rating that assesses a property’s overall energy performance, based on national data to control for differences among climate, building uses, and operations. A score of 50 represents the national median.

  - N.B. : from 0 to 100. Median = 50, Good = more than 75. 
  - documentation : https://www.energystar.gov/buildings/benchmark/analyze_benchmarking_results#what

- Remarks : 
  - The modern SI unit for heat energy is the joule (J); one Btu equals about 1055 J (varying within the range 1054–1060 J depending on the specific definition; source : wikipedia).
  - 1 kBtu = 1.055 MJ
  - 1 kWh = 3.412 kBtu = 3.6 MJ
  - sf = square foot / 1 m² = 10.76391 sf / 1 sf = 0.092903 m²
  - 1 therm = 99.98 kBtu.

- Remark EUI :
  - EUI : Energy Use Intensity (Energy in kBtu/sf)
  - EUI documentation : https://www.energystar.gov/buildings/benchmark/understand_metrics/what_eui
  - Generally, a low EUI signifies good energy performance. However, certain property types will always use more energy than others. For example, an elementary school uses relatively little energy compared to a hospital.


- SiteEUI(kBtu/sf) : Site Energy Use Intensity (EUI) is a property's Site Energy Use divided by its gross floor area. Site Energy Use is the annual amount of all the energy consumed by the property on-site, as reported on utility bills. Site EUI is measured in thousands of British thermal units (kBtu) per square foot.

- SiteEUIWN(kBtu/sf) : Weather Normalized (WN) Site Energy Use Intensity (EUI) is a property's WN Site Energy divided by its gross floor area (in square feet). WN Site Energy is the Site Energy Use the property would have consumed during 30-year average weather conditions. WN Site EUI is measured in measured in thousands of British thermal units (kBtu) per square foot.

- SourceEUI(kBtu/sf) : Source Energy Use Intensity (EUI) is a property's Source Energy Use divided by its gross floor area. Source Energy Use is the annual energy used to operate the property, including losses from generation, transmission, & distribution. Source EUI is measured in thousands of British thermal units (kBtu) per square foot.

- SourceEUIWN(kBtu/sf) : Weather Normalized (WN) Source Energy Use Intensity (EUI) is a property's WN Source Energy divided by its gross floor area. WN Source Energy is the Source Energy Use the property would have consumed during 30-year average weather conditions. WN Source EUI is measured in measured in thousands of British thermal units (kBtu) per square foot.

- SiteEnergyUse(kBtu) : The annual amount of energy consumed by the property from all sources of energy.

- SiteEnergyUseWN(kBtu) : The annual amount of energy consumed by the property from all sources of energy, adjusted to what the property would have consumed during 30-year average weather conditions.

- SteamUse(kBtu) : The annual amount of district steam consumed by the property on-site, measured in thousands of British thermal units (kBtu).

- Electricity(kWh) : The annual amount of electricity consumed by the property on-site, including electricity purchased from the grid and generated by onsite renewable systems, measured in kWh.

- Electricity(kBtu) : The annual amount of electricity consumed by the property on-site, including electricity purchased from the grid and generated by onsite renewable systems, measured in thousands of British thermal units (kBtu).

- NaturalGas(therms) : The annual amount of utility-supplied natural gas consumed by the property, measured in therms.

- NaturalGas(kBtu) : The annual amount of utility-supplied natural gas consumed by the property, measured in thousands of British thermal units (kBtu).

#### CO2 emissions : 

- TotalGHGEmissions : The total amount of greenhouse gas emissions, including carbon dioxide, methane, and nitrous oxide gases released into the atmosphere as a result of energy consumption at the property, measured in metric tons of carbon dioxide equivalent. This calculation uses a GHG emissions factor from Seattle CIty Light's portfolio of generating resources. This uses Seattle City Light's 2015 emissions factor of 52.44 lbs CO2e/MWh until the 2016 factor is available. Enwave steam factor = 170.17 lbs CO2e/MMBtu. Gas factor sourced from EPA Portfolio Manager = 53.11 kg CO2e/MBtu.

- GHGEmissionsIntensity : Total Greenhouse Gas Emissions divided by property's gross floor area, measured in kilograms of carbon dioxide equivalent per square foot. This calculation uses a GHG emissions factor from Seattle City Light's portfolio of generating resources


### Other remarks :

ENERGY STAR Portfolio Manager — the Industry Standard for Benchmarking Commercial Buildings.

Portfolio Manager is an interactive resource management tool that enables you to benchmark the energy use of any type of building, all in a secure online environment. Nearly 25% of U.S. commercial building space is already actively benchmarking in Portfolio Manager, making it the industry-leading benchmarking tool. It also serves as the national  benchmarking tool in Canada.


