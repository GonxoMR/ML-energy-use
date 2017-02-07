# Machine Learning to forecast energy consumptiona and production.
This project is based on the emoncms.org users and their REST API. 

The machine in which this project is develop runs a Debian 8.7
The code is built in Python 2.7.9 with numpy 1.8.2 and pandas 0.14.1.

## Obtaining_data.py
In order to get the data from them, a dictionary with the apikeys their type of source and the if of the feeds is necesary. It is store into .csv called 'apiKeyDictionary.csv' with three columns "key", "type" and "id" like showed in Table 1.


| key     | type              | id  |
|---------|-------------------|-----|
| APIKEY1 | house_consumption | ID1 |
| APIKEY2 | grid_power        | ID2 |
| APIKEY3 | solar_power       | ID3 |
*Table 1. Example of a apiKeyDictionary.csv structure*

