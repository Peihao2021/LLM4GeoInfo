import os
from dotenv import load_dotenv
from openai import OpenAI
import geopy.distance
from geoc.geocoding import GeoCoding
import pandas as pd
import re

load_dotenv()

def clean_string(s):
    return re.sub(r'[^0-9.]', '', s)

def query(message):
    '''this is a simple function for querying chat-gpt'''
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
        model="gpt-4")

    return chat_completion.choices[0].message.content

def zero_shot(df):
    '''This code is to test 0-shot training for the gpt-4 model'''
    df_new = df.copy()
    df_new['Predicted Latitude'] = None
    df_new['Predicted Longitude'] = None

    for _, row in df_new.iterrows():
        city = row['AccentCity']
        country_initials = row['Country']
        message = f"The geo-coordinates of {city}, {country_initials} are"
        response = query(message)
        print(response)

        # error catching
        if len(response) > 30:
            continue

        predicted_lat, predicted_lng = response.split(',')

        predicted_lat = clean_string(predicted_lat)
        predicted_lng = clean_string(predicted_lng)

        row['Predited Latitude'] = predicted_lat
        row['Predicted Longitude'] = predicted_lng
    
    return df_new


def createCitiesList():
    '''This code is directly sourced from https://github.com/prabin525/spatial-llm/blob/main/spatial-awareness/create_cities_list.py'''
    in_file_loc = 'places_with_lat_lng_within_continent.txt'
    geocoder = GeoCoding()
    places = []
    with open(in_file_loc) as f:
        for each in f.readlines():
            splitted = each.split("\t")
            lat, lng, _, _, _ = geocoder.get_lat_lng(splitted[0])
            places.append({
                'name': splitted[0],
                'lat': lat,
                'lng': lng
            })
    for each in places:
        places2 = places.copy()
        places2.remove(each)
        for i, c in enumerate(places2):
            if i == 0:
                dis = geopy.distance.distance(
                    [each['lat'], each['lng']], [c['lat'], c['lng']]
                ).km
                near_city = c['name']
                near_dis = dis
                far_city = c['name']
                far_dis = dis
            else:
                dis = geopy.distance.distance(
                    [each['lat'], each['lng']], [c['lat'], c['lng']]
                ).km
                if dis < near_dis:
                    near_dis = dis
                    near_city = c['name']
                if dis > far_dis:
                    far_dis = dis
                    far_city = c['name']
        each['near_city'] = near_city
        each['far_city'] = far_city

    df = pd.DataFrame(places)
    df.to_json('us_cities.json')

