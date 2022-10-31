# Cartwright Github Pages

# Cartwright

Cartwirght categorizes spatial and temporal features in a dataset. 

Cartwright uses natural language processing and heuristic functions to determine the correct categorization of a feature. The goal of this project was for a given dataframe where we expect some kind of geospatial and temporal columns, automatically infer:

-   Country
-   Admin levels (0 through 3)
-   Timestamp (from arbitrary formats)
-   Latitude
-   Longitude
-   Dates (including format)
-   Time resolution for date columns


 The model and transformation code can be used locally by installing the pip package or downloaded the github repo and following the directions found in /training_model/README.md.

## Install Cartwright 
First you need to have a python 3.8 or higher install in your env. 

    pip install cartwright    
   
Once it is installed you can run it from a command line

	python -m cartwright.categorize --path "your/path/to/a/csvfile.csv"
 This command will run Cartwright over the csv file and return the categorizations for each feature.
 You can also import Cartwright into your python code.
	 

    from cartwright import categorize
	Cartwright=categorize.CartwrightClassify()
    

 
 Cartwright's CartwrightClassify class takes in some init parameters which are: 
	

 - **model_version** - defaults to the latest 
 -  **number_of_samples**- This specifies how many values to sample from each column, defaults to 100 
 - **seconds_to_finish** - This specifies the max duration of the analysis, defaults to 40 seconds.

	 
We instantiate Cartwright with the number of random samples (default=100) you want to take from each column of your csv or df. In most cases more samples of each column will result is more accurate classifications, however it will increases execution time. 


 ### cartwright.columns_classified(path)
  The main function is ***columns_classified***. This function returns an array that classifies each column in our csv/df by using a combination of the predictions from the model along with validation code for each classification.

    preds=Cartwright.columns_classified(path="your/path/to/a/csvfile.csv")
    print(preds)

  Possible information returned for each column are:
  1. **'column'**:Column name in csv
  2. **'classification'**: an array with a few possible values:
      - *'category'*: The final classified name for the column** most important return
      - *'subcategory'*: This will be returned if there is addition sub categories for the classification. E.g. [{'Category': 'Geo', 'type': 'Latitude (number)'}]
      - *'format'*: If the column is classified as a date it will give the best guess at the format for that date column. 
      - *'time_resolution'*: for temporal columns, the automatically detected density of timestamps in this column.
      - *'parser'*: This lets you know what parser was used for determining if it was a valid date or not. The two options are 'Arrow' and 'Util' which represent the arrow and dateutil libraries respectively.
      - *'match_type'*: How the classification was made. LSTM for the model, fuzzy for fuzzywuzzy match on column headers.

5. **'fuzzyColumn'**: This is returned if the column name is similar enough to any word in a list of interest. Shown below.
    [  
    "Date",  
    "Datetime",  
    "Timestamp",  
    "Epoch",  
    "Time",  
    "Year",  
    "Month",  
    "Lat",  
    "Latitude",  
    "lng",  
    "Longitude",  
    "Geo",  
    "Coordinates",  
    "Location",  
    "location",  
    "West",  
    "South",  
    "East",  
    "North",  
    "Country",  
    "CountryName",  
    "CC",  
    "CountryCode",  
    "State",  
    "City",  
    "Town",  
    "Region",  
    "Province",  
    "Territory",  
    "Address",  
    "ISO2",  
    "ISO3",  
    "ISO_code",  
    "Results",  
]
   
The 'format' of a date classification are created using this reference sheet: https://strftime.org/ . 

## Under the hood
The workflow consists of four main sections. 
1. Cartwright model
2. Heuristic functions
3. Column header fuzzy match
4. Automatic temporal resolution detection

Workflow overview

![Alt text](geotime_classify/resources/cartwright2.png?raw=true "Workflows")

