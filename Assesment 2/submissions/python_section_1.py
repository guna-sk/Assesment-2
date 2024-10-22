from typing import Dict, List

import pandas as pd

def reverse_by_n(lst, n):
    result = []
    for i in range(0, len(lst), n):
        group = []
        for j in range(min(n, len(lst) - i)):
            group.insert(0, lst[i + j])
        result.extend(group)
    return result
print(reverse_by_n([1,2,3,4,5,6,7,8],3))
print(reverse_by_n([1,2,3,4,5],2))
print(reverse_by_n([10,20,30,40,50,60,70],4))

def group_by_length(strings):
    length_dict = {}
    for string in strings:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    return dict(sorted(length_dict.items()))

input_list1 = ["apple", "bat", "car", "elephant", "dog", "bear"]
input_list2 = ["one", "two", "three", "four"]

print(group_by_length(input_list1))
print(group_by_length(input_list2))



def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                list_key = f"{new_key}[{i}]"
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, list_key, sep=sep).items())
                else:
                    items.append((list_key, item))
        else:
            items.append((new_key, v))
    return dict(items)

# Example usage:
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

flattened_dict = flatten_dict(nested_dict)
print(flattened_dict)



from itertools import permutations

def unique_permutations(nums):
    return [list(p) for p in set(permutations(nums))]
input_list = [1, 1, 2]
result = unique_permutations(input_list)
print(result)



import re

def find_all_dates(text):
    date_pattern = r'\b\d{2}-\d{2}-\d{4}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{4}\.\d{2}\.\d{2}\b'
    return re.findall(date_pattern, text)

text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
dates = find_all_dates(text)
print(dates)

import polyline
from math import radians, cos, sin, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def decode_polyline_to_df(polyline_str):
    coords = polyline.decode(polyline_str)
    df = pd.DataFrame(coords, columns=['latitude', 'longitude'])
    df['distance'] = 0.0
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i - 1, 'latitude'], df.loc[i - 1, 'longitude']
        lat2, lon2 = df.loc[i, 'latitude'], df.loc[i, 'longitude']
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)
    return df

polyline_str = 'u{~vFvyys@lCzAjBdArA'
df = decode_polyline_to_df(polyline_str)
print(df)



import numpy as np

def rotate_90_clockwise(matrix):
    return [list(reversed(col)) for col in zip(*matrix)]

def transform_matrix(matrix):
    n = len(matrix)
    transformed = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            row_sum = sum(matrix[i]) - matrix[i][j]  
            col_sum = sum(matrix[k][j] for k in range(n)) - matrix[i][j]
            transformed[i][j] = row_sum + col_sum
    return transformed
def rotate_and_transform(matrix):
    rotated_matrix = rotate_90_clockwise(matrix)
    return transform_matrix(rotated_matrix)

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_transform(matrix)
print(result)


import numpy as np

def check_time_completeness(df):
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    df.set_index(['id', 'id_2'], inplace=True)
    result = pd.Series(index=df.index.unique(), dtype=bool)
    full_week_seconds = 7 * 24 * 60 * 60
    for index in result.index:
        group = df.loc[index]
        total_seconds = np.sum((group['end_datetime'] - group['start_datetime']).dt.total_seconds())
        result[index] = total_seconds < full_week_seconds

    return result

df = pd.read_csv('dataset-1.csv')
result = check_time_completeness(df)
print(result)
