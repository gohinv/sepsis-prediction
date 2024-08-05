import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def plot_dataset(columns, data, n_cols):  
    # Define the size of the grid
    n_rows = (len(columns) + n_cols - 1) // n_cols  # Calculate the required number of rows

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 4, n_rows * 3))  # Adjust the size as needed

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Plot a histogram on each subplot
    for ax, column in zip(axes, columns):
        ax.hist(data[column].dropna(), bins=30, alpha=0.7, color='blue')  # Drop NA for clean histogram plotting
        ax.set_title(column)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    # Hide any unused subplots if you have an uneven number of columns
    for ax in axes[len(columns):]:
        ax.axis('off')

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Display the plot
    plt.show()

def custom_impute(column):
    # Convert the series to a numpy array for easier manipulation
    values = column.values
    mean_value = np.nanmean(values)  # Compute mean ignoring NaN
    
    # Identify the index of the first non-NaN value
    first_valid_index = column.first_valid_index()
    
    # If the first entry itself is NaN, we need to handle initial NaN values
    if first_valid_index is not None:
        # Fill NaN values up to the first valid index with the mean
        values[:first_valid_index] = np.where(np.isnan(values[:first_valid_index]), mean_value, values[:first_valid_index])
        
    # Create a Series from the values to utilize pandas' fillna method for forward filling
    return pd.Series(values, index=column.index).fillna(method='ffill')

def differential_features(dataset, columns):
    temp_data = np.array(dataset)
    for column in columns:
        data = np.array(dataset[column])
        nanpos = np.where(~np.isnan(data))[0]
        diff = data.copy().astype(float)
        if len(nanpos) <= 1:
            diff[:] = np.nan
            temp_data = np.column_stack((temp_data, diff))
        else:
            diff[:nanpos[1]] = np.nan
            for p in range (1, len(nanpos)-1):
                diff[nanpos[p] : nanpos[p+1]] = data[nanpos[p]] - data[nanpos[p-1]]
            diff[nanpos[-1]:] = data[nanpos[-1]] - data[nanpos[-2]]
            temp_data = np.column_stack((temp_data, diff))
    return temp_data

def slide_window(data_arr, col_idx):
    """
    Calculate dynamic statistics in a six-hour sliding window
    :param data_arr: data after using a forward-filling strategy
    :param col_idx: selected features
    :return: time-series features
    """
    data = data_arr[:, col_idx]
    max_values = [[0 for col in range(len(data))]
                  for row in range(len(col_idx))]
    min_values = [[0 for col in range(len(data))]
                  for row in range(len(col_idx))]
    mean_values = [[0 for col in range(len(data))]
                   for row in range(len(col_idx))]
    median_values = [[0 for col in range(len(data))]
                   for row in range(len(col_idx))]
    std_values = [[0 for col in range(len(data))]
                   for row in range(len(col_idx))]
    diff_std_values = [[0 for col in range(len(data))]
                   for row in range(len(col_idx))]

    for i in range(len(data)):
        if i < 6:
            win_data = data[0:i + 1]
            for ii in range(6 - i):
                win_data = np.row_stack((win_data, data[i]))
        else:
            win_data = data[i - 6: i + 1]

        for j in range(len(col_idx)):
            dat = win_data[:, j]
            if len(np.where(~np.isnan(dat))[0]) == 0:
                max_values[j][i] = np.nan
                min_values[j][i] = np.nan
                mean_values[j][i] = np.nan
                median_values[j][i] = np.nan
                std_values[j][i] = np.nan
                diff_std_values[j][i] = np.nan
            else:
                max_values[j][i] = np.nanmax(dat)
                min_values[j][i] = np.nanmin(dat)
                mean_values[j][i] = np.nanmean(dat)
                median_values[j][i] = np.nanmedian(dat)
                std_values[j][i] = np.nanstd(dat)
                diff_std_values[j][i] = np.std(np.diff(dat))

    win_features = list(chain(max_values, min_values, mean_values,
                              median_values, std_values, diff_std_values))
    win_features = (np.array(win_features)).T

    return win_features

def feature_extract(dataset):
    labels = np.array(dataset['SepsisLabel'])
    data_prime = dataset.drop(columns=['Bilirubin_direct', 'TroponinI', 'Fibrinogen', 'SepsisLabel'])
    # obtain np array of dataset stacked with differential features
    data_with_diff = differential_features(data_prime, vitals + labvals)
    # 6 hour sliding window for ['HR', 'O2Sat', 'SBP', 'MAP', 'Resp'] = [0, 1, 3, 4, 6] in vitals
    print(data_with_diff.shape)
    # data_slide_win = slide_window(data_with_diff, [0, 1, 3, 4, 6])
    # features = np.column_stack((data_with_diff, data_slide_win))
    features = data_with_diff
    return features, labels

def process_data(dataset):
    features, labels = feature_extract(dataset)
    index = [i for i in range(len(labels))]
    np.random.shuffle(index)
    features = features[index]
    labels = labels[index]
    return features, labels
    