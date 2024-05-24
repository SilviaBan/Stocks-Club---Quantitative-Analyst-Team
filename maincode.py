import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def analyze_stock(ticker, start_date, end_date):
    # Fetch historical data
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')

    # Calculate Up (1) or Down (0) movements
    df['Change'] = df['Close'].diff()
    df['Up'] = (df['Change'] > 0).astype(int)
    df['Down'] = (df['Change'] <= 0).astype(int)

    # Define sequences dictionary to include all specified sequences
    sequences = {
        'UUUUD': [1, 1, 1, 1, 0],
        'UUUDD': [1, 1, 1, 0, 0],
        'UUDUU': [1, 1, 0, 1, 1],
        'UUDUD': [1, 1, 0, 1, 0],
        'UDUDU': [1, 0, 1, 0, 1],
        'UDUDD': [1, 0, 1, 0, 0],
        'UDDUU': [1, 0, 0, 1, 1],
        'UDDDD': [1, 0, 0, 0, 0],
        'DUUUD': [0, 1, 1, 1, 0],
        'DUUDD': [0, 1, 1, 0, 0],
        'DUDUU': [0, 1, 0, 1, 1],
        'DUDUD': [0, 1, 0, 1, 0],
        'DUDDD': [0, 1, 0, 0, 0],
        'DDUDU': [0, 0, 1, 0, 1],
        'DDDUU': [0, 0, 0, 1, 1],
        'DDDUD': [0, 0, 0, 1, 0],
        'DDDDU': [0, 0, 0, 0, 1],
        'DDDDD': [0, 0, 0, 0, 0],
    }

    # Dynamically check for sequences
    for name, seq in sequences.items():
        df[name] = 0
        for i in range(len(seq), len(df)):
            match = True
            for j in range(len(seq)):
                if (seq[j] == 1 and df['Up'].iloc[i-len(seq)+j] != 1) or (seq[j] == 0 and df['Down'].iloc[i-len(seq)+j] != 1):
                    match = False
                    break
            if match:
                df.loc[df.index[i], name] = 1  # Use .loc to avoid SettingWithCopyWarning

    # Consolidate signals (1 on days to enter based on any matching sequence)
    df['Signal'] = df[list(sequences.keys())].sum(axis=1).clip(upper=1)

    # Assuming buying at the close of a signal day and selling at the close of the next day
    df['Positions'] = df['Signal'].shift(1)

    # Calculate returns
    df['Market Returns'] = df['Close'].pct_change()
    df['Strategy Returns'] = df['Market Returns'] * df['Positions']

    # Calculate Predicted and Actual Directions
    df['Predicted Direction'] = np.where(df['Strategy Returns'] > 0, 1, 0)
    df['Actual Direction'] = np.where(df['Market Returns'] > 0, 1, 0)

    # Determine if the prediction was correct
    df['Correct Prediction'] = (df['Predicted Direction'] == df['Actual Direction']).astype(int)

    # Split data into training and test sets
    train_size = int(0.7 * len(df))
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    # Calculate accuracy for training set
    train_accuracy = train_df['Correct Prediction'].sum() / train_df['Correct Prediction'].count() * 100

    # Calculate accuracy for test set
    test_accuracy = test_df['Correct Prediction'].sum() / test_df['Correct Prediction'].count() * 100

    # Calculate cumulative returns for training set
    train_df['Cumulative Market Returns'] = (1 + train_df['Market Returns']).cumprod()
    train_df['Cumulative Strategy Returns'] = (1 + train_df['Strategy Returns']).cumprod()

    # Calculate cumulative returns for test set
    test_df['Cumulative Market Returns'] = (1 + test_df['Market Returns']).cumprod()
    test_df['Cumulative Strategy Returns'] = (1 + test_df['Strategy Returns']).cumprod()

    # Plotting the results for training set
    plt.figure(figsize=(14, 7))
    plt.plot(train_df.index, train_df['Cumulative Market Returns'], label='Market Returns (Train)', color='blue')
    plt.plot(train_df.index, train_df['Cumulative Strategy Returns'], label='Strategy Returns (Train)', color='green')
    plt.title('Market Returns vs Strategy Returns (Training Set)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    # Plotting the results for test set
    plt.figure(figsize=(14, 7))
    plt.plot(test_df.index, test_df['Cumulative Market Returns'], label='Market Returns (Test)', color='blue')
    plt.plot(test_df.index, test_df['Cumulative Strategy Returns'], label='Strategy Returns (Test)', color='green')
    plt.title('Market Returns vs Strategy Returns (Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    # Print Accuracy
    print(f"Direction Prediction Accuracy (Training): {train_accuracy:.2f}%")
    print(f"Direction Prediction Accuracy (Test): {test_accuracy:.2f}%")

# Example usage
analyze_stock('BA', '2020-03-20', '2024-03-21')
