# Coiled technical exercise

This take-home assignment involves processes and analyzing the
[NYC Taxi and Limousine Commission (TLC)](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
dataset. Below we outline the steps to download the dataset to your
local machine and present a few tasks to work on.

Please don't worry about developing a perfect solution for each task. Go through
each task, work on some kind of solution, and then iterate on these solutions if
time permits. We anticipate this entire assignment should take a few hours of time total.

# Download dataset

You can download a zip file with the TLC dataset from [this Google Drive link](https://drive.google.com/file/d/1uZeqdCj5XippV2vt8InZ2whwf1DwRjHR/view?usp=sharing).

*NOTE: the uncompressed version of this dataset takes up ~9 GiB of disk space.*

# Assignment

After the TLC dataset has been downloaded, create a Python script or
Jupyter notebook for each of the tasks below.

## Task 1

Using Dask, load the dataset, clean any non-sensical outliers, and make a few plots
to explore the dataset. Note that Dask itself doesn't have plotting utilities, so
feel free to use `matplotlib`, `bokeh`, or whatever plotting library you prefer.

## Task 2

Using Dask, load the dataset, clean any non-sensical outliers, and save the dataset
to Parquet format. 

## Task 3

Imagine the steps in Task 2 are a daily computational task. The outputted Parquet
dataset feeds data science workloads that get accessed dozens of times throughout
the day. Can you characterize the performance of this CSV-to-parquet processing?
How can it be improved? How should we optimize this workflow?

## Task 4

Using the Parquet dataset, train an XGBoost model to predict the rider tip
percentage from other values in the dataset. How well does the model perform?

## Task 5 (Optional)

Repeat Task 4, but instead of running Dask on your local machine, use a Coiled cluster
to run Dask on AWS. Note that the CSV-version of the TLC dataset is publicly available
using this S3 glob string: `s3://nyc-tlc/trip data/yellow_tripdata_2019-*.csv`.