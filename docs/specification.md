# Project Specification

## Data

Data contains records from shops for 2 years in a daily granulation of every product.

Raw Files:
 * sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
 * test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
 * sample_submission.csv - a sample submission file in the correct format.
 * items.csv - supplemental information about the items/products.
 * item_categories.csv  - supplemental information about the items categories.
 * shops.csv- supplemental information about the shops.

## Goal

Predicting total amount of specific items sold by a shop in a month 
(shop, item) -> amount

## Metric

Goal is to minimize Root Mean Squared Error (RMSE) 
Additionally: True classes are clipped to [0,20] values



