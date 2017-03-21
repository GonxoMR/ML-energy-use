# Machine Learning to forecast energy consumptiona and production.
This project is based on the emoncms.org users and their REST API. 

The machine in which this project is develop runs a Debian 8.7
The code is built in Python 2.7.9 with numpy 1.8.2 and pandas 0.14.1.

## Obtaining_data.py
In order to get the data from them, a dictionary with the apikeys their type of source and the if of the feeds is necesary. It is store into .csv called 'apiKeyDictionary.csv'. Originally it has four columns "key", "type" and "id" like showed in Table 1.


| key     | type              | id  |lat_long  |
|---------|-------------------|-----|----------|
| APIKEY1 | house_consumption | ID1 |lat1,long1|
| APIKEY2 | grid_power        | ID2 |lat2,long2|
| APIKEY3 | solar_power       | ID3 |lat3,long3|
*Table 1. Example of a apiKeyDictionary.csv structure*

After running Obtaining_data.py, it will add two additional columns to the apiKeyDictionary.csv. These are "start" and "end" this are the initial and final times of the feed. This is later used in the file Obtaining_weather_data.py to call to weather underground over the relevant dates to obtain weather data. 
