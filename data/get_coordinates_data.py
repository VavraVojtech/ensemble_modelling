import pandas as pd
import numpy as np
import logging
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import requests

def get_map_cze_okresy():
    # Load the GeoJSON Data from GitHub
    url_okres = "https://raw.githubusercontent.com/siwekm/czech-geojson/master/okresy.json" # czech_republic, kraje, okresy, obce, okresky, 

    # Fetch the GeoJSON data from the URL
    response = requests.get(url_okres)
    data = response.json()

    # Load GeoJSON into a GeoDataFrame
    okresy_shapes = gpd.GeoDataFrame.from_features(data['features'])
    okresy_codes = [feature['nationalCode'] for feature in data['features']]
    okresy_names = [feature['name'] for feature in data['features']]

    okresy_shapes['name'] = okresy_names
    okresy_shapes['state_abbreviation'] = okresy_codes

    okresy_shapes['longitude'] = okresy_shapes.centroid.x
    okresy_shapes['latitude'] = okresy_shapes.centroid.y
    return okresy_shapes

def get_map_cze_kraje():
    # Load the GeoJSON Data from GitHub
    url_kraj = "https://raw.githubusercontent.com/siwekm/czech-geojson/master/kraje.json" # czech_republic, kraje, okresy, obce, okresky, 

    # Fetch the GeoJSON data from the URL
    response = requests.get(url_kraj)
    data = response.json()

    # Load GeoJSON into a GeoDataFrame
    kraje_shapes = gpd.GeoDataFrame.from_features(data['features'])
    kraje_codes = [feature['nationalCode'] for feature in data['features']]
    kraje_names = [feature['name'] for feature in data['features']]

    kraje_shapes['name'] = kraje_names
    kraje_shapes['state_abbreviation'] = kraje_codes

    kraje_shapes['longitude'] = kraje_shapes.centroid.x
    kraje_shapes['latitude'] = kraje_shapes.centroid.y
    return kraje_shapes

def okresy_map(df):
    # Create the plot with boundaries and consumption data
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    df.boundary.plot(ax=ax, linewidth=0.8)
    ax.scatter(df['longitude'], df['latitude'], color='chocolate', s=10, label='Střed okresu')

    plt.title(f"Středy (cenroidy) jednotlivých okresů")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    # plt.grid()
    plt.legend()
    return plt

def kraje_map(df):
    # Create the plot with boundaries and consumption data
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    df.boundary.plot(ax=ax, linewidth=0.8)
    ax.scatter(df['longitude'], df['latitude'], color='chocolate', s=10, label='Střed kraje')

    plt.title(f"Středy (cenroidy) jednotlivých okresů")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    # plt.grid()
    plt.legend()
    return plt

def run_map_df(draw_maps, nuts='okresy'):

    if nuts == 'okresy':
        shapes = get_map_cze_okresy()
        if draw_maps:
            file_name = 'output_map/map_okresy.png'
            okresy_map_ = okresy_map(shapes)
            okresy_map_.savefig(file_name)
            logging.info(f'Map saved to {file_name}')
    elif nuts == 'kraje':
        shapes = get_map_cze_kraje()
        if draw_maps:
            file_name = 'output_map/map_kraje.png'
            kraje_map_ = kraje_map(shapes)
            kraje_map_.savefig(file_name)
            logging.info(f'Map saved to {file_name}')

    return shapes

if __name__ == "__main__":
    # this code is for obtaining geocoordinates for czech republic, addicional functions are present for drawing maps
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S",
                        format='%(asctime)s %(levelname)s | %(message)s')

    draw_maps = True
    nuts = 'okresy'          # ['kraje', 'okresy']

    df = run_map_df(draw_maps, nuts)
    file_name = f'output_data/map_data_{nuts}.csv'
    df.to_csv(file_name, sep=';', index=False)
    logging.info(f'Data saved to {file_name}')









    





