import requests
import datetime
import numpy as np
import pandas as pd

# Set up the destination directory
dataDir = 'DATA_DIRECTORY'

# User Apikey and feed ids are neccesary to get the feed's data.
# Writting and reading Apikeys are needed in case there is not a reading apikey bypass. 
# Direct to 'apiKeyDictionary.csv' location.
secretsDir = 'SECRETS_DIRECTORY'
apiDic = pd.read_csv(secretsDir+'/apiKeyDictionary.csv')
types = np.unique(apiDic['type'])

# File with security secrets stored
secrets = pd.read_json(secretsDir+'/secrets.json')

# Creating a saving file
store = pd.HDFStore(dataDir+'/'+str(datetime.datetime.now())+'_feeds.h5')

# Looping for all types of feed
for type in types:

	for index, row in apiDic.loc[(apiDic['type']==type),['key','id']].iterrows():
		
		# NOTE: adding troubleshooting might be interesting. 
		# Obtaining streamed data
		response = requests.get("https://emoncms.org/feed/export.json?apikey="+str(row['key'])+"&id="+str(row['id'])+"&start=0&CCBYPASS="+secrets['BYPASS'][0])
		
		# Obtaining meta data for error detection and time generation
		meta = eval(requests.get("https://emoncms.org/feed/timevalue.json?id="+str(row['id'])+"&apikey="+str(row['key'])).content)

		# Checking data consistancy in bytes
		if len(response.content) % 4 == 0:
			
			# Creating a numpy array from data steam. Decodification is done here.
			array = np.fromstring(response.content, 'float32')
			
			# Finding starting time point. 
			# It is computed as the time value of the last observation minus the size of the data
			end_point = int(meta['time'])
			start_point = end_point - array.size*10
			
			# Checking if the obtained data stream has the same value as the last observation in the server. 
			if array[-1] != float(meta['value']):
				print("Error: The last value of the array do not match the last value obtained from the feed: "+str(row['id']))
	
			# Giving time to the data observations
			df = pd.DataFrame({"unit":array},
						   index= pd.date_range(pd.to_datetime(start_point+10, unit='s')
						   ,pd.to_datetime(end_point, unit='s'), freq='10S'))
								
		else:
			print("Error: The number of elements recieved is not multiple of 4. Decodification is not posible. Feed: "+ str(row['id']))
		
		# Saving individual data frames into the hf5 file. Each feed is individually saved with its type.
		store[str(type)+'_'+str(row['id'])] = df