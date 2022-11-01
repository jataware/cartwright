
# Cartwright
![Tests](https://github.com/jataware/cartwright/actions/workflows/tests.yml/badge.svg)

Cartwirght categorizes spatial and temporal features in a dataset. 

Cartwright uses natural language processing and heuristic 
functions to determine the best guess categorization of a feature. 
The goal of this project was for a given dataframe where we expect
some kind of geospatial and temporal columns, automatically infer:

-   Country
-   Admin levels (0 through 3)
-   Timestamp (from arbitrary formats)
-   Latitude
-   Longitude
-   Dates (including format)
-   Time resolution for date columns


 The model and transformation code can be used locally by installing
 the pip package or downloaded the github repo and following the directions
 found in /docs.

# Simple use case

Cartwright has the ability to classify features of a dataframe which can help
with automation tasks that normally require a human in the loop.
For a simple example we have a data pipeline that ingests dataframes and
creates a standard timeseries plots or a map with datapoints. The problem is these new dataframes
are not standarized, and we have no way of knowing which columns contain dates or locations data.
By using Cartwright we can automatically infer which columns are dates or coordinate values and 
continue with our pipeline.

Here is the dataframe with :

| x_value  |  y_value   | date_value | Precip |
|:---------|:----------:|-----------:|--------|
| 7.942658 | 107.240322 | 07/14/1992 | .2     |
| 7.943745 | 137.240633 | 07/15/1992 | .1     |
| 7.943725 | 139.240664 | 07/16/1992 | .3     |


python code example and output.
    
    from cartwright import categorize
    cartwright = categorize.CartwrightClassify()
    categorizations = cartwright.columns_categorized(path="path/to/csv.csv")
    for column, category in categorization.items():
        print(column, category)

You can see from the output we were able to infer that x_value and y_values were geo category with subcategory of latitude and longitude. In some cases these can be impossible to tell apart since all latitude values are valid longitude values. For our date feature the category is time and the subcategory is date. The format is correct and we were able to pick out the time resolution of one day.  


    x_value {'category': <Category.geo: 'geo'>, 'subcategory': <Subcategory.latitude: 'latitude'>, 'format': None, 'time_resolution': {'resolution': None, 'unit': None, 'density': None, 'error': None}, 'match_type': [<Matchtype.LSTM: 'LSTM'>], 'fuzzyColumn': None}
    
    y_value {'category': <Category.geo: 'geo'>, 'subcategory': <Subcategory.longitude: 'longitude'>, 'format': None, 'time_resolution': {'resolution': None, 'unit': None, 'density': None, 'error': None}, 'match_type': [<Matchtype.LSTM: 'LSTM'>], 'fuzzyColumn': None}

    date_value {'category': <Category.time: 'time'>, 'subcategory': <Subcategory.date: 'date'>, 'format': '%m/%d/%Y', 'time_resolution': {'resolution': TimeResolution(uniformity=<Uniformity.PERFECT: 1>, unit=<TimeUnit.day: 86400.0>, density=1.0, error=0.0), 'unit': None, 'density': None, 'error': None}, 'match_type': [<Matchtype.LSTM: 'LSTM'>], 'fuzzyColumn': None}

    precip_value {'category': None, 'subcategory': None, 'format': None, 'time_resolution': {'resolution': None, 'unit': None, 'density': None, 'error': None}, 'match_type': [], 'fuzzyColumn': None}

With this information we can now convert the date values to a timestamp and plot a timeseries with other features.

