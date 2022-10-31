# Geotime Classify

![Tests](https://github.com/jataware/geotime_classify/actions/workflows/tests.yml/badge.svg)

This model is a recurrent neural network that uses LSTM to learn text classification. The goal of this project was for a given spreadsheet where we expect some kind of geospatial and temporal columns, can we automatically infer things like:

-   Country
-   Admin levels (0 through 3)
-   Timestamp (from arbitrary formats)
-   Latitude
-   Longitude
-   Which column likely contains the "feature value"
-   Which column likely contains a modifier on the  `feature`

 The model and transformation code can be used locally by installing the pip package or downloaded the github repo and following the directions found in /training_model/README.md.

## Pip install geotime classify
First you need to install numpy, scipy, pandas, joblib, pip, torch and torchvision.

    pip install -r requirements.txt

  or 

    conda install -c conda-forge numpy
    conda install -c conda-forge scipy
    conda install -c conda-forge scikit-learn
    conda install -c conda-forge matplotlib
    conda install -c conda-forge pandas
    conda install -c conda-forge joblib
    conda install -c conda-forge pip
    conda install pytorch torchvision cpuonly -c pytorch
    
    
    
Now you can pip install the geotime_classify repo. To pip install this repo use:

    pip install geotime-classify
 
 
Once it is installed you can instantiate the geotime_classify with the number of random samples (n) you want to take from each column of your csv. To take 100 samples from each column run. In most cases more samples of each column will result is more accurate classifications, however it will increase the time of processing. 

    from geotime_classify import geotime_classify as gc
    GeoTimeClass = gc.GeoTimeClassify(1000)

Now we have our GeoTimeClassify class instantiated we can use the ***columns_classified*** function. 

   
 ### geotime_classify.columns_classified(path)
  The next function is ***columns_classified***. This function returns an array that classifies each column in our csv by using a combination of the predictions from the LSTM model along with validation code for each classification. 
  

    c_classified=GeoTimeClass.columns_classified('pathtocsv')
    print(c_classified)

  Possible information returned for each column are:
  1. **'column'**:Column name in csv
  2. **'classification'**: an array with a few possible values:
      - *'category'*: The final classified name for the column** most important return
      - *'subcategory'*: This will be returned if there is addition sub categories for the classification. E.g. [{'Category': 'Geo', 'type': 'Latitude (number)'}]
      - *'format'*: If the column is classified as a date it will give the best guess at the format for that date column. 
      - *'time_resolution'*: for temporal columns, the automatically detected density of timestamps in this column.
      - *'parser'*: This lets you know what parser was used for determining if it was a valid date or not. The two options are 'Arrow' and 'Util' which represent the arrow and dateutil libraries respectively.
      - *'dayFirst'*: A boolean value. If the column is classified as a date the validation code will try to test if day or month come first in the format, which is necessary for accurate date standardization.
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
   
For the classification data there are a set number of possible classifications after the validation code. dayFirst can be 'True' or 'False'.
The 'format' of a date classification are created using this reference sheet: https://strftime.org/ . 
Possible classifciation options: 
1. "category": "None"
2. "category": "geo", "subcategory":"continent"
3. "category": "geo", "subcategory":"country_name"
4. "category": "geo", "subcategory":"state_name"
5. "category": "geo", "subcategory":"city_name"
6. "category": "geo", "subcategory":"ISO3"
7. "category": "geo", "subcategory":"ISO2"
12. "category": "geo", "subcategory": "longitude"
13. "category": "geo", "subcategory": "latitude"
19. "category": "unknown date" , "subcategory": None
14. "category": "time", "subcategory": "date", "format": format, "Parser": "Util",  "DayFirst": dayFirst
14. "category": "time", "subcategory": "date", "format": format, "Parser": "arrow",  "DayFirst": dayFirst
51. "category": "Boolean"


## Under the hood
The workflow consists of four main sections. 
1. Geotime Classify Model
2. Heuristic Functions
3. Column Header Fuzzy Match
4. Automatic Temporal Resolution Detection


Workflow overview

![Alt text](training_model/images/GeoTime_ClassifyWorkflow.png?raw=true "Workflows")


## Geotime Classify Model
This model is a type recurrent neural network that uses LSTM to learn text classification. The model is trained on Fake data provided by [Faker](https://faker.readthedocs.io/en/master/). The goal was for a given spreadsheet where we expect some kind of geospatial and temporal columns, can we automatically infer things like:

-   Country
-   Admin levels (0 through 3)
-   Timestamp (from arbitrary formats)
-   Latitude
-   Longitude
-   Which column likely contains the "feature value"
-   Which column likely contains a modifier on the  `feature`

To do this, we collected example data from Faker along with additional locally generated data. The model was built using pytorch. We used padded embedding, and LSTM cell, a linear layer and finally a LogSoftmax layer. This model was trained with a dropout of .2 to reduce overfitting and improving model performance. 

	    self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)
        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout_layer = nn.Dropout(p=0.2)
After a few iterations the model was performing well enough with accuracy hovering around 91 percent with 57 categories.
Confusion Matrix:

  ![Alt text](training_model/images/confusionMatrix.png?raw=true "Confusion Matrix")

Now the model was able to ingest a string and categorize it into one the 57 categories. 

## The Heuristic functions
The heuristic functions ingest the prediction classifications from the model along with the original data for  validation tests. If the data passes the test associated with the classification the final classification is made and returned. If it failed it will return 'None' or 'Unknown Date' if the model classified the column as a date. If addition information is needed for future transformation of the data these functions try to capture that. For example if a column is classified as a Date the function will try validate the format and return it along with the classification.

## Column Header Fuzzy Match
This is the most simple part of the workflow. For each column header we try to match that string to a word of interest. If there is a high match ratio the code returns the word of interest. For more info you can see Fuzzywuzzy docs [here](https://pypi.org/project/fuzzywuzzy/).


## Automatic Temporal Resolution Detection
If a dataset contains data at evenly spaced intervals, the resolution is automatically detected according to the following process:
1. convert all unique dates/times to unix timestamps
2. sort the timestamps and compute the delta time between each
3. find the median of the deltas
4. characterize the uniformity of the deltas
    - if all deltas are identical (to within some small ϵ), mark as `PERFECT`
    - if the maximum deviation from the median is less than 1% the magnitude of the median, mark as (approximate) `UNIFORM`
    - otherwise mark as `NOT_UNIFORM`
5. if the deltas are perfectly or approximately uniform, find the closest matching time unit from a preset list:
    - ~~`millisecond` (1e-3 * second)~~
    - `second` (1)
    - `minute` (60 * second)
    - `hour` (60 * minute)
    - `day` (24 * hour)
    - `week` (7 * day)
    - `year` (365 * day)
    - `month` (year / 12)
    - `decade` (10 * year + 2 * day)
    - `century` (100 * year + 24 * day)
    - `millennium` (1000 * year + 242 * day)

    in the future, temporal units will be drawn from a more comprehensive units ontology
6. convert the median delta to a proportion of the matched unit, and set as the temporal unit for the dataset

Currently milliseconds may experience issues due to floating point precision errors, and thus may not be detected by this process.

time resolutions are represented by a `TimeResolution` object with values: `uniformity` enum, `unit` enum, `density` value (in the unit given), and mean `error` value. If the detection process fails, the object will be `None`



## Retraining geotime_classify with github repo
To get started read the README in the training_model directory. 

## Building the `pip` package

```
bump2version --current-version=XYZ patch setup.py
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*
```
