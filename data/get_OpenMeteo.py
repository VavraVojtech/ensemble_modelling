import pandas as pd
import pickle
from datetime import timedelta
import logging
import time
import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import openmeteo_requests
import requests_cache
from retry_requests import retry

from data.get_coordinates_data import run_map_df


def retrieve_OpenMeteo_data(type, params, name, print_info=True):
	# https://open-meteo.com/en/terms
	# Less than 10'000 API calls per day, 5'000 per hour and 600 per minute.

	# Setup the Open-Meteo API client with cache and retry on error
	if type == 'history':
		url = "https://archive-api.open-meteo.com/v1/archive"
		expire_after = -1
	elif type == 'forecast':
		url = "https://api.open-meteo.com/v1/forecast"
		expire_after = 3600
	elif type == 'history_forecast':
		url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
		expire_after = 3600
	elif type == 'forecast_ensamble':
		url = "https://ensemble-api.open-meteo.com/v1/ensemble"
		expire_after = 3600

	cache_session = requests_cache.CachedSession('.cache', expire_after = expire_after)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

	params_strict = {
    	"hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    	           "cloud_cover", "visibility", "wind_speed_10m", "wind_gusts_10m", "direct_radiation"], # , "sunshine_duration", "direct_radiation"],
    	"timezone": "Europe/Berlin"
	}
	params.update(params_strict)

	# Make sure all required weather variables are listed here
	# The order of variables in hourly or daily is important to assign them correctly below
	responses = openmeteo.weather_api(url, params=params)

	# Process location + for-loop for multiple locations or weather models
	dataframes_dict = {}
	
	for i, name in enumerate(name):
		response = responses[i]
		if print_info:
			print(f"Okres {i}: {name}")
			print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
			print(f"Elevation {response.Elevation()} m asl")
			# print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
			# print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

		# Process hourly data. The order of variables needs to be the same as requested.
		hourly = response.Hourly()
		hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
		hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
		hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
		hourly_apparent_temperature = hourly.Variables(3).ValuesAsNumpy()
		hourly_cloud_cover = hourly.Variables(4).ValuesAsNumpy()
		hourly_visibility = hourly.Variables(5).ValuesAsNumpy()
		hourly_wind_speed_10m = hourly.Variables(6).ValuesAsNumpy()
		hourly_wind_gusts_10m = hourly.Variables(7).ValuesAsNumpy()
		hourly_direct_radiation = hourly.Variables(8).ValuesAsNumpy()

		hourly_data = {"date": pd.date_range(
			start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
			end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
			freq = pd.Timedelta(seconds = hourly.Interval()),
			inclusive = "left"
		)}
		hourly_data["temperature_2m"] = hourly_temperature_2m
		hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
		hourly_data["dew_point_2m"] = hourly_dew_point_2m
		hourly_data["apparent_temperature"] = hourly_apparent_temperature
		hourly_data["cloud_cover"] = hourly_cloud_cover
		hourly_data["visibility"] = hourly_visibility
		hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
		hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m
		hourly_data["direct_radiation"] = hourly_direct_radiation

		hourly_dataframe = pd.DataFrame(data = hourly_data)
		hourly_dataframe['date'] = hourly_dataframe['date'].dt.tz_convert('Europe/Berlin')
		dataframes_dict[name] = hourly_dataframe
	
	return dataframes_dict


def get_openmeteo(df, start_date, end_date):
	# Setup the Open-Meteo API client with cache and retry on error
	weather_openmeteo = {}
	iteration_count = 0

	# this filter is only for testing purposes
	# df = df[df['name'].isin(['Hlavní město Praha', 'Hodonín', 'Brno-venkov'])]

	logging.info(f"Processing {len(df)} nuts...")
	for i in range(1):					# there is no need for split so far for retrieving data

		# OpenMeteo --- History
		type = 'history'
		name = df['name']
		params = {
				"latitude": df['latitude'],
				"longitude": df['longitude'],
				"start_date": start_date,
				"end_date": end_date
			}	
		weather_openmeteo = retrieve_OpenMeteo_data(type, params, name)
		# weather_openmeteo[name]['date'] = weather_openmeteo[name]['date'].dt.tz_convert('Europe/Prague')
		logging.info(f"Okres {name} processed sucessfully.")

		# API call control
		# Less than 10'000 API calls per day, 5'000 per hour and 600 per minute.
		iteration_count += 1
		if iteration_count >= 16:       # there is no need for split so far for retrieving data
			print("Reached 16 iterations, waiting for 1 minute...")
			time.sleep(60)  # Wait for 1 minute (60 seconds)
			iteration_count = 0  # Reset counter after waiting
    
    
	file_path = os.path.join(project_root, f'output_data/raw_weather_data_OpenMeteo_{start_date}_{end_date}.pkl')
	with open(file_path, 'wb') as file:
		pickle.dump(weather_openmeteo, file)
	logging.info(f"Dictionary saved to {file_path}")

	return weather_openmeteo

def aggregate_weather_data(weather_data):
    logging.info("Aggregating weather data (calculating mean across all districts)...")
    
    all_dfs = []
    for name, df in weather_data.items():
        temp_df = df.copy()
        if 'date' in temp_df.columns:
            temp_df.set_index('date', inplace=True)
        all_dfs.append(temp_df)

    combined_df = pd.concat(all_dfs)
    aggregated_df = combined_df.groupby(combined_df.index).mean()
    aggregated_df.reset_index(inplace=True)
    
    return aggregated_df

def test_Praha():

	logging.basicConfig(stream=sys.stdout, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S",
	                 format='%(asctime)s %(levelname)s | %(message)s')
	lat = 50.088
	lon = 14.4208
	type = 'forecast_ensamble'
	name = ['Praha']
	params = {
				"latitude": lat,
				"longitude": lon,
				"models": "icon_seamless"
			}	
	
	df_test = retrieve_OpenMeteo_data(type, params, name, print_info=True)
	print(df_test)
	print(df_test['Praha'].columns)

	corr_matrix = df_test['Praha'].corr()
	sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
	plt.title('Correlation Matrix hour')
	plt.show()

	df_ = df_test['Praha'].groupby(df_test['Praha']['date'].dt.date).mean()
	corr_matrix = df_.corr()
	sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
	plt.title('Correlation Matrix day')
	plt.show()
	return df_test

if __name__ == "__main__":
     
	logging.basicConfig(stream=sys.stdout, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S",
	                 format='%(asctime)s %(levelname)s | %(message)s')
	
	# control retrieving data
	load_map_df = False
	nuts = 'okresy'		# ['okresy', 'kraje']
	load_OpenMeteo_data = False
	save_to_csv = True
	
	# clients are promised to be our customers in range (1 month befor today sounds good)
	today = pd.to_datetime('today').normalize()
	zkz_start_date = pd.to_datetime(today - timedelta(days=30)).normalize()
	zkz_end_date = today

	# load MND map info
	if load_map_df:
		logging.info('Loading map data from file...')
		df_nuts = pd.read_csv(f'input_data/map_data_{nuts}.csv', sep=';')
	else:
		df_nuts = run_map_df(draw_maps = True, nuts = nuts)

	start_date = pd.to_datetime('2016-06-01').normalize()
	end_date   = pd.to_datetime('2025-01-10').normalize()
	start_date = start_date.strftime('%Y-%m-%d')
	end_date = end_date.strftime('%Y-%m-%d')

	if load_OpenMeteo_data:
		file_path = f'input_data/raw_weather_data_OpenMeteo_{start_date}_{end_date}.pkl' # 'input_data/weather_data_meteostat.pkl'
		weather_data = pd.read_pickle(file_path)
	else: 
		weather_data = get_openmeteo(df_nuts, start_date, end_date)

	# print(weather_data)

	# get weights based on consumption
	df_nuts.drop('geometry', axis=1, inplace=True)

	# get weighted (based on consumtion) weather distributed over okresy
	# prumer pro celou CR -> hodinove udaje v radcich
	avg_df = aggregate_weather_data(weather_data)
	avg_df['date'] = pd.to_datetime(avg_df['date'])
	avg_df = avg_df.set_index('date')

	print(avg_df)

	# Save to CSV
	if save_to_csv:
		file_name = 'output_data/OpenMeteo_train_data_.csv'
		avg_df.to_csv(file_name)
		logging.info(f'Data saved to {file_name}')
	
	# Resample data from 00:00 to 23:59 for daily average
	# REGULAR DAY
	daily_avg_0_24 = avg_df.resample('D').mean()
	daily_avg_0_24.index = daily_avg_0_24.index.strftime('%Y-%m-%d')
	daily_avg_0_24 = daily_avg_0_24.iloc[1:-1]
	if save_to_csv:
		file_name_0 = 'output_data/OpenMeteo_train_data_0_24_.csv'
		daily_avg_0_24.to_csv(file_name_0)
		logging.info(f'Data saved to {file_name_0}')

	# Resample data from 06:00 to 05:59 for daily average
	# GAS DAY
	daily_avg_6_6 = avg_df.resample('D', offset='6H').mean()
	daily_avg_6_6.index = daily_avg_6_6.index.strftime('%Y-%m-%d')
	daily_avg_6_6 = daily_avg_6_6.iloc[2:-1]
	if save_to_csv:
		file_name_6 = 'output_data/OpenMeteo_train_data_6_6_.csv'
		daily_avg_6_6.to_csv(file_name_6)
		logging.info(f'Data saved to {file_name_6}')

	# # save history to pickle
	# file_path = os.path.join(project_root, f'output_data/weather_data_meteostat_{start_date}_{end_date}_.pkl')
	# with open(file_path, 'wb') as file:
	# 	pickle.dump(history, file)
	# print(f"Dictionary saved to {file_path}")






	# OpenMeteo --- Future
	# type = "forecast"
	# lat = [50.075539]
	# lon = [14.4378]
	# name = ['test1']	
	# start_date = pd.to_datetime('2024-11-30').normalize()
	# end_date = pd.to_datetime('2024-12-06').normalize() + timedelta(days=6)
	# start_date = start_date.strftime('%Y-%m-%d')
	# end_date = end_date.strftime('%Y-%m-%d')	
	# params = {
	#         "latitude": lat,
	#         "longitude": lon,
	#         "start_date": start_date,
	#         "end_date": end_date
	#     }	
	# forecast = retrieve_OpenMeteo_data(type, params, name)
	# print(forecast)
