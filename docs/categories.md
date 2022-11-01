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
 '%Y-%m-%d', <br>
 '%Y_%m_%d', <br>
 '%Y/%m/%d', <br>
 '%Y.%m.%d', <br>
 '%Y%m%d', <br>
 '%Y-%m-%d %H:%M:%S', <br>
 '%Y/%m/%d %H:%M:%S', <br>
 '%Y_%m_%d %H:%M:%S', <br>
 '%Y.%m.%d %H:%M:%S', <br>
 '%d-%m-%Y', <br>
 '%d/%m/%Y %H:%M:%S', <br>
 '%d_%m_%Y %H:%M:%S', <br>
 '%d.%m.%Y %H:%M:%S', <br>
 '%d-%m-%y', <br>
 '%d_%m_%Y', <br>
 '%d_%m_%y',<br>
 '%d/%m/%Y', <br>
 '%d/%m/%y', <br>
 '%d.%m.%Y', <br>
 '%d.%m.%y',<br>
 '%d-%m-%Y %H:%M:%S', <br> 
 '%a, %d %b %Y', <br>
 '%A, %B %d, %Y',<br>
 '%A, %B %d, %Y, %H:%M:%S', <br> 
 '%d %B %Y', <br>
 '%d %B %y', <br>
 '%B %d, %Y', <br>
 '%m/%d/%y %H:%M:%S %p', <br>
 '%m-%d-%Y', <br>
 '%m/%d/%Y %H:%M:%S', <br>
 '%m_%d_%Y %H:%M:%S', <br>
 '%m.%d.%Y %H:%M:%S', <br>
 '%m-%d-%y', <br>
 '%m_%d_%Y', <br>
 '%m_%d_%y', <br>
 '%m/%d/%Y', <br>
 '%m/%d/%y', <br>
 '%m.%d.%Y', <br>
 '%m.%d.%y',<br>
 '%m-%d-%Y %H:%M:%S', <br> 
 '%Y%d', <br>
 '%Y-%m',  <br>
 '%Y/%m', <br>
 '%Y.%m',<br>
 '%Y_%m', <br>
 '%Y-%m-%dT%H:%M:%S', <br> 
 'unix_time', <br>
 '%d', <br>
 '%A', <br>
 '%a', <br>
 '%m', <br>
 '%B', <br>
 '%b', <br>
 '%Y'
