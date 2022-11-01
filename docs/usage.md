---
layout: default
title: Usage
nav_order: 3
has_toc: true
---
# Usage

There are two ways to run the model categorization. The first way is to load Cartwright into python.

  from cartwright import categorize
	cartwright = categorize.CartwrightClassify(seconds_to_finish=40)
  # you can also pass in a pandas df if you already have one loaded in you environment
  categorizations = cartwright.columns_categorized(path="path/to/csv.csv", df=None)

When instantiating the CartwrightClassify class you can set model_version, number_of_samples, and seconds_to_finish. model_version will read in the model you are interested in running by default it will be the lastest offical build. number_of_samples will be the number of values samples from each column. Seconds_to_finish sets the max amount of time the analysis should run. At 40 seconds any columns not yet classified will be skipped. 

You can also run Cartwright from the command line with: 

  python3 -m cartwright.categorize path="path/to/csv.csv" --num_samples 100 

The path needs to be set to the correct location of the csv file you want to categorize. 
The num_samples flag will sample that many values from each column. The large the sample size the longer the run can take. 
This will always run the default model version so if you build your own you will need to change that manualy.


