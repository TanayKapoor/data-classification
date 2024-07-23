import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def handle_file(file_path):
    df = pd.read_csv(file_path)
    
    numerical_data = df.select_dtypes(include=[np.number])
    non_numerical_data = df.select_dtypes(exclude=[np.number])

    for col in numerical_data.columns:
        print(f"Ueda profiling for {col}")
        print(f"Mean: {numerical_data[col].mean()}")
        print(f"Median: {numerical_data[col].median()}")
        print(f"Standard deviation: {numerical_data[col].std()}")
        print(f"Min: {numerical_data[col].min()}")
        print(f"Max: {numerical_data[col].max()}")
        print(f"25th percentile: {numerical_data[col].quantile(0.25)}")
        print(f"75th percentile: {numerical_data[col].quantile(0.75)}")
        print(f"Number of missing values: {numerical_data[col].isna().sum()}")
        print(f"Number of unique values: {numerical_data[col].nunique()}")
        print(f"Number of zeros: {numerical_data[col].apply(lambda x: 1 if x == 0 else 0).sum()}")
        print("\n")
    
    detect_outliners(numerical_data)

    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Ueda Profiling')

    for i, col in enumerate(numerical_data.columns):
        ax = axes[i//2, i%2]
        ax.plot(numerical_data[col])
        ax.set_title(f"{col} line graph")
        ax.axhline(numerical_data[col].mean(), color='r', linestyle='dashed', linewidth=1)
        ax.axhline(numerical_data[col].median(), color='g', linestyle='dashed', linewidth=1)
        ax.axhline(numerical_data[col].quantile(0.25), color='b', linestyle='dashed', linewidth=1)
        ax.axhline(numerical_data[col].quantile(0.75), color='y', linestyle='dashed', linewidth=1)
        ax.legend(['Mean', 'Median', '25th percentile', '75th percentile'])
        # mark outliners
        q1 = numerical_data[col].quantile(0.25)
        q3 = numerical_data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        ax.axhline(lower_bound, color='black', linestyle='dashed', linewidth=1)
        ax.axhline(upper_bound, color='black', linestyle='dashed', linewidth=1)
        ax.legend(['Mean', 'Median', '25th percentile', '75th percentile', 'Lower bound', 'Upper bound'])

    plt.show()

    calculate_z_score(numerical_data)
    print(numerical_data.describe())

    # plot z-score for each numerical column in a new figure
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Z-Score')

    for i, col in enumerate(numerical_data.columns):
        ax = axes[i//2, i%2]
        ax.plot(numerical_data[col + '_zscore'])
        ax.set_title(f"{col} z-score")
        ax.axhline(0, color='r', linestyle='dashed', linewidth=1)
        ax.axhline(1, color='g', linestyle='dashed', linewidth=1)
        ax.axhline(-1, color='g', linestyle='dashed', linewidth=1)
        ax.legend(['Mean', '1', '-1'])  
    plt.show()


def detect_outliners(df):
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        print(f"Outliners for {col}: {df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]}")
    
def calculate_z_score(df):
    for col in df.columns:
        df[col + '_zscore'] = (df[col] - df[col].mean())/df[col].std()
    return df

    
handle_file("data/archive/mbl_dataset/sport_location_20240311160816.csv")