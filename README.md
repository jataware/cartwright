# Geotime Classify

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
First you need to install numpy, torch and torchvision.

    pip install numpy 
    pip install scipy
    pip install pandas
    pip install joblib
    pip install torch
    pip install torchvision

  or 

    conda install -c conda-forge numpy
    conda install -c conda-forge scipy
    conda install -c conda-forge pandas
    conda install -c conda-forge joblib
    conda install pytorch torchvision cpuonly -c pytorch
    
    
    
Now you can pip install the geotime_classify repo. To pip install this repo use:

    pip install geotime_classify
 
 
Once it is installed you can instantiate the geotime_classify with the number of random samples (n) you want to take from each column of your csv. To take 100 samples from each column run. In most cases more samples of each column will result is more accurate classifications, however it will increase the time of processing. 

    import geotime_classify 
    geotime=geotime_classify.GeoTimeClassify(100)

Now we have our GeoTimeClassify class instantiated we can use the functions available. The first one and most basic is the ***predictions*** function.
### geotime_classify.predictions(path)
This returns the geotime_classify model predictions for each of the columns in the csv. 

    preds=geotime.predictions('pathtocsv')
    print(preds)

You should see an array with a dict for each column. The keys in each dict are 
1. **'column'**: which is the name of the column in the csv you provided,
2. **'values'**: these are the random sampled values from each column that were used. There will be the same number of these as the number n you instantiated the class with.
3. **'avg_predictions'**: an array which contains:
   
     -'averaged tensor', This is an averaged tensor created by calculating the mean from (n) prediction tensors that the model created.    
     -'averaged_top_category', Returns the category with the highest match for our averaged tensor.
4. **'model_predictions'**: Contains the raw model outputs as prediction tensors for each value that was randomly sampled from the column. 
   
 ### geotime_classify.columns_classified(path)
  The next function is ***columns_classified***. This function returns an array that classifies each column in our csv by using a combination of the predictions from the LSTM model along with validation code for each classification. 
  

    c_classified=geotime.columns_classified('pathtocsv')

  Possible information returned for each column are:
  1. **'column'**:Column name in csv
  2. **'classification'**: an array with a few possible values:
	  A. *'Category'*: The final classified name for the column** most important return
	  B. *'type'*: This will be returned if there is addition sub categories for the classification. E.g. [{'Category': 'Geo', 'type': 'Latitude (number)'}]
	  C. *'Format'*: If the column is classified as a date it will give the best guess at the format for that date column. 
	  D. *'Parser'*: This lets you know what parser was used for determining if it was a valid date or not. The two options are 'Arrow' and 'Util' which represent the arrow and dateutil libraries respectively. 
	  E. *'DayFirst'*: A boolean value. If the column is classified as a date the validation code will try to test if day or month come first in the format, which is necessary for accurate date standardization. 
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
 ### geotime_classify.add_iso8601_columns(path, formats='default')

Lastly, there is ***add_iso8601_columns***. This function returns a dataframe with added columns for each column that was classified as 'Date'. Each new date column will be named iso8601_*.  * will be replaced with the index of the original column. Formats should be set to 'default' unless you want the output to be in a different valid format. An example might be '%B %d, %Y'.

        df=geotime.add_iso8601_columns('pathtocsv', formats='default')

This function would return Dataframe 1 as Datframe 2

Dataframe 1

 index | Date | Feature 
--|--|--
| 1 | July, 8th 2020 | 1 |
| 2 | July, 9th 2020 | 2 |

Dataframe 2.

 index | Date | Feature | iso8601_1 
--|--|--|--
| 1 | July, 8th 2020 | 1 | 2020-07-08 |
| 2 | July, 9th 2020 | 2 | 2020-07-09 |
 

## Under the hood
The workflow consists of four main sections. 
1. Geotime Classify Model
2. Heuristic Functions
3. Column Header Fuzzy Match
4. Date standardization 

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

## Date Standardization
This is a big challenge, but pairing the model classification along with pre-built libraries like [arrow](https://arrow.readthedocs.io/en/latest/) and [dateutil](https://dateutil.readthedocs.io/en/stable/index.html) this functionality works on most common date formats. The key to using these libraries to standardize dates is knowing the format the date is currently in to transform it. For this the geotime_classify model is able to classify the date format well enough to know which library to use for parsing. For certain classifications a heuristic function is use to determine if day comes before month in the current format. If day does come first that is passed to the parser which then can correctly transforms the date. Once the date is in a standardized form you can transform it again to any valid format you want. This functionality is exposed by the *add_iso8601_columns* function, where you can pass any valid date format to the formats parameter. 



## Retraining geotime_classify with github repo
To get started read the README in the training_model directory. 

