---
layout: default
title: Supported Categories
nav_order: 5
has_toc: true
---
# Supported Categories

There are only a limited set of supported categories, but new categories can be added easily. 

Currently the main categories that we support fall into spatial and temporal categories. 

For spatial categories we can categorize at a subcategory level latitude, longitude, country name, ISO2, ISO3, state, and continent. You can see each category for spatial values in the geos.py file in the categories dir.

For temporal categoreis we can categorize year, month, month days, day, day name, timespans and many traditional date formats. These can be seen in the categories timespan, dates, and partial_dates files. We use python strftime formats as our standard. 

Date formats support are:
 '%Y-%m-%d', 
 '%Y_%m_%d', 
 '%Y/%m/%d', 
 '%Y.%m.%d', 
 '%Y%m%d', 
 '%Y-%m-%d %H:%M:%S', 
 '%Y/%m/%d %H:%M:%S', 
 '%Y_%m_%d %H:%M:%S', 
 '%Y.%m.%d %H:%M:%S',
 '%d-%m-%Y',
 '%d/%m/%Y %H:%M:%S', 
 '%d_%m_%Y %H:%M:%S',
 '%d.%m.%Y %H:%M:%S', 
 '%d-%m-%y', 
 '%d_%m_%Y', 
 '%d_%m_%y',
 '%d/%m/%Y', 
 '%d/%m/%y', 
 '%d.%m.%Y', 
 '%d.%m.%y',
 '%d-%m-%Y %H:%M:%S', 
 '%a, %d %b %Y', 
 '%A, %B %d, %Y',
 '%A, %B %d, %Y, %H:%M:%S', 
 '%d %B %Y', 
 '%d %B %y', 
 '%B %d, %Y',
 '%m/%d/%y %H:%M:%S %p', 
 '%m-%d-%Y', 
 '%m/%d/%Y %H:%M:%S',
 '%m_%d_%Y %H:%M:%S', 
 '%m.%d.%Y %H:%M:%S', 
 '%m-%d-%y', 
 '%m_%d_%Y',
 '%m_%d_%y', 
 '%m/%d/%Y', 
 '%m/%d/%y', 
 '%m.%d.%Y', 
 '%m.%d.%y',
 '%m-%d-%Y %H:%M:%S', 
 '%Y%d', 
 '%Y-%m', 
 '%Y/%m', 
 '%Y.%m',
 '%Y_%m',
 '%Y-%m-%dT%H:%M:%S', 
 'unix_time',
 '%d', 
 '%A', 
 '%a', 
 '%m', 
 '%B', 
 '%b', 
 '%Y'