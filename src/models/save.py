import pandas as pd
from content_based_filtering import transform_data, save_transformed_data
from data_cleaning import data_for_content_filtering

filtered_data = pd.read_csv('data/collab_filtered_data.csv')

prepared_data = data_for_content_filtering(filtered_data)

transformed_data = transform_data(prepared_data)

save_transformed_data(transformed_data, 'data/transformed_hybrid_data.npz')

