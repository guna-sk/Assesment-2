import pandas as pd

import numpy as np

def calculate_distance_matrix(file_path):
    df = pd.read_csv(file_path)
    distance_df = df.pivot_table(index='Location_A', columns='Location_B', values='Distance', fill_value=np.inf)
    locations = distance_df.index.union(distance_df.columns)
    distance_matrix = pd.DataFrame(np.inf, index=locations, columns=locations)
    for row in distance_df.itertuples():
        distance_matrix.at[row.Location_A, row.Location_B] = row.Distance
    np.fill_diagonal(distance_matrix.values, 0)
    for k in locations:
        for i in locations:
            for j in locations:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]
    distance_matrix = distance_matrix.combine(distance_matrix.T, np.minimum)

    return distance_matrix



def unroll_distance_matrix(distance_matrix):
    unrolled_data = []

    for start in distance_matrix.index:
        for end in distance_matrix.columns:
            if start != end:  
                unrolled_data.append({'id_start': start, 'id_end': end, 'distance': distance_matrix.at[start, end]})

    unrolled_df = pd.DataFrame(unrolled_data)
    
    return unrolled_df



def find_ids_within_ten_percentage_threshold(unrolled_df, reference_id):
    reference_distances = unrolled_df[unrolled_df['id_start'] == reference_id]['distance']
    average_distance = reference_distances.mean()
    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1
    within_threshold = unrolled_df[
        (unrolled_df['distance'] >= lower_bound) & 
        (unrolled_df['distance'] <= upper_bound)
    ]['id_start'].unique()
    return sorted(within_threshold)



def calculate_toll_rate(unrolled_df):
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    for vehicle, coefficient in rate_coefficients.items():
        unrolled_df[vehicle] = unrolled_df['distance'] * coefficient
    
    return unrolled_df


import datetime

def calculate_time_based_toll_rates(toll_df):
    time_ranges = [
        (datetime.time(0, 0), datetime.time(10, 0), 0.8),  
        (datetime.time(10, 0), datetime.time(18, 0), 1.2), 
        (datetime.time(18, 0), datetime.time(23, 59, 59), 0.8) 
    ]
    weekend_discount = 0.7
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    expanded_data = []
    for _, row in toll_df.iterrows():
        for day in days_of_week:
            for start_time in pd.date_range('00:00:00', '23:59:59', freq='1H'):
                start_time = start_time.time()
                end_time = (datetime.datetime.combine(datetime.date.today(), start_time) + datetime.timedelta(hours=1)).time()
                
                if day in days_of_week[:5]:  
                    factor = 1.0
                    for start_range, end_range, discount in time_ranges:
                        if start_range <= start_time < end_range:
                            factor = discount
                            break
                else:  
                    factor = weekend_discount
                vehicle_rates = {vehicle: row[vehicle] * factor for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']}
                expanded_data.append({
                    'id_start': row['id_start'],
                    'id_end': row['id_end'],
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    **vehicle_rates
                })

    return pd.DataFrame(expanded_data)

