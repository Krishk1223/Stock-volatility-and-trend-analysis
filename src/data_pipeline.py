import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
import src.config as cf


def fetch_stock_data(symbols, start_date, end_date, intervals, adjust=True):
    """
    Step 1: Data Retrieval
    Fetch historical stock data for given symbols and date range.
    
    Parameters:
        symbols (list): List of stock ticker symbols.
        start_date (datetime): Start date in 'YYYY-MM-DD' format.
        end_date (datetime): End date in 'YYYY-MM-DD' format.
        intervals (str): Data interval (e.g., '1d' for daily).
        adjust (bool): Whether to adjust data for corporate actions like dividends and splits. Defaults to True.
    
    Returns:
        data (pd.DataFrame): DataFrame containing the Opening and Closing (adjusted) stock data.
    """
    try:
        data = yf.download(symbols, interval=intervals, start=start_date, end=end_date, auto_adjust=adjust)[['Open', 'Close']]
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        sys.exit(1)
    
    if data.empty:
        print("No data fetched, please check symbols and date range.")
        sys.exit(1)
    
    required_cols = ['Open', 'Close']
    if not all(col in data.columns for col in required_cols):
        print(f"Data fetched is missing required columns. Expected Open and Close columns. Please check data")
        sys.exit(1)
    
    return data

def log_returns(data, price_col='Close', drop=False):
    """
    Calculate log returns for given stock data (and drops NaN values). Useful for long term returns + 
    volatility calculations and used in time series modelling as well as Monte Carlo simulations.
    Parameters:
        data (pd.DataFrame): DataFrame containing stock data with 'Close' price column.
        price_col (str): Column name for price data to calculate log returns on. Defaults to 'Close'.
                            Options ['Open', 'Close']
        drop (bool) : whether to drop NaN values resulting from log return calculation. Defailts to false
                      to preserve data size for any further checks.
    Returns:
        log_returns (pd.DataFrame): DataFrame containing log returns of the stock prices.
    """
    if price_col not in ['Open', 'Close']:
        print(f"Invalid price column specified. Allowed options are 'Open' and 'Close'. Defaulting to 'Close'.")
        price_col = 'Close'

    log_returns = np.log(data[price_col] / data[price_col].shift(1))
    if drop:
        log_returns = log_returns.dropna()
    
    return log_returns

def check_trading_days(data, calendar=cf.STOCK_EXCHANGE_TIMEZONE):
    """
    Check if the data contains only trading days based on the specified calendar. Used in Validation step.
    This check doesn't serve much purpose as yfinance should only return trading days if you use other data 
    sources then this is useful.

    Parameters:
        data (pd.DataFrame): DataFrame containing stock data to be checked.
        calendar (str): Calendar to use for checking trading days. Defaults to 'NYSE'.
                        See pandas_market_calendars documentation for available calendar options.
                        
    Returns:
        trading_data (pd.DataFrame): DataFrame containing only rows corresponding to trading days. 
    """
    try:
        calendar = mcal.get_calendar(calendar)
    except Exception as e:
        print(f"Error fetching calendar: {e}. Defaulting to NYSE calendar.")
        calendar = mcal.get_calendar('NYSE')

    trading_days = calendar.valid_days(start_date=data.index.min(), end_date=data.index.max())
    trading_data = data[data.index.isin(trading_days)]
    return trading_data

def missing_data_checks(data):
    """
    Check for missing values in the dataset. Used in Validation step.
    Parameters:
        data (pd.DataFrame): DataFrame containing stock data to be checked.
    Returns:
        missing (bool): True if missing values are detected, False otherwise.
    """
    missing = False
    if data.isna().sum().any():
        print("Missing values detected in following columns: ")
        print(data.isna().sum())
        missing = True
    else:
        print("No missing values detected.")
    
    return missing

def duplicate_data_checks(data):
    """
    Check for duplicate rows in the dataset. Used in Validation step.
    Parameters:
        data (pd.DataFrame): DataFrame containing stock data to be checked.
    Returns:
        duplicates (bool): True if duplicate rows are detected, False otherwise.
    """
    duplicates = False
    if data.index.duplicated().any():
        print("Duplicate dates detected.")
        duplicates = True
    else:
        print("No duplicate dates detected.")

    return duplicates

def negative_price_checks(data):
    """
    Checks for any negative stock prices in dataset. Used in Validation Step.
    Parameters:
        data (pd.DataFrame): DataFrame containing stock data to be checked.
        mode (str): Operation mode. Either 'validation' or 'clean'
    Returns:
        negative_prices (bool): True if negative prices are detected, False otherwise.
        negative_price_rows (pd.DataFrame): DataFrame containing rows with negative prices. If none then returns empty df.
    """
    negative_prices = False
    negative_price_mask = (data['Open'] < 0) | (data['Close'] < 0)
    negative_price_rows = data[negative_price_mask.any(axis=1)]
    if not negative_price_rows.empty:
        negative_prices = True
        print("Negative prices detected")
        return negative_prices, negative_price_rows
    else:
        print("No negative prices detected.")
        return negative_prices, pd.DataFrame()

def close_open_anomalies(data, threshold_theta=None):
    """
    Checks for any major anomalies between close price of one day and open price of next day above a certain
    threshold and if so flags the rows where open price of next day is above the threshold. This is to account
    for any major off market trades affecting the overall trend through sharp changes.
    Used in Validation Step.
    Parameters:
        data (pd.DataFrame): DataFrame containing stock data to be checked.
        threshold_theta (int): Threshold for detecting anomalies as a multiplier of standard deviation
                                (if not specified it will default to value in config.py).
    Returns:
        close_open_anomalies (bool): True if anomalies are detected, False otherwise.
        anomaly_rows (pd.DataFrame, None): DataFrame containing rows with anomalies. If none then returns empty df.
    """


    new_data = data.copy()
    anomalies = False
    threshold_theta = threshold_theta if threshold_theta is not None else cf.ANOMALY_THETA_THRESHOLD

    # Used a rolling mean of 21 days and std to calculate the threshold for anomaly detection
    # focused on log space gap rather than percentage change.
    gap = np.log(new_data['Open'] / new_data['Close'].shift(1))
    mean_gap = gap.rolling(window=cf.ROLLING_WINDOW, min_periods=cf.MIN_WINDOW).mean()
    std_gap = gap.rolling(window=cf.ROLLING_WINDOW, min_periods=cf.MIN_WINDOW).std()
    anomaly_upper_threshold = mean_gap + threshold_theta * std_gap
    anomaly_lower_threshold = mean_gap - threshold_theta * std_gap

    #clip the gap to handle anomalies without dropping data.
    gap_clipped = gap.clip(lower=anomaly_lower_threshold, upper=anomaly_upper_threshold)

    clipped_open = new_data['Close'].shift(1) * (np.exp(gap_clipped))

    #first any reduces across stock rows and 2nd one checks if any column has true.
    anomalies = (np.abs(new_data['Open'] - clipped_open) > 1e-6).any().any() # turns true if any anomalies are detected

    return anomalies, clipped_open

def validate_data(data):
    """
    Step 2: Data Validation
    Perform Sanity Checks on data to ensure quality and reliability.
    Checks include:
        - Missing values
        - Duplicate rows
        - Negative stock prices
        - Close to next day open anomalies (e.g close to next day open price change above 5%)
    Parameters:
        data (pd.DataFrame): Cleaned stock data to be validated.
    Returns:
        checks_dict (dict): Dictionary containing results of all validation checks with keys:
            missing (bool): True if missing values are detected, False otherwise.
            duplicates (bool): True if duplicate rows are detected, False otherwise.
            negative (bool): True if negative prices are detected, False otherwise.
            negative_data (pd.DataFrame): DataFrame containing rows with negative prices. If none then returns empty df.
            close_open_anomalies (bool): True if close-open anomalies are detected, False otherwise.
            clipped_open (pd.Series): Series containing clipped open prices for close-open anomaly detection.
    """
    
    #run all checks and then store result in a dictionary for further use in cleaning step.
    missing = missing_data_checks(data)
    duplicates = duplicate_data_checks(data)
    negative, negative_data = negative_price_checks(data)
    anomalies, clipped_open = close_open_anomalies(data)

    checks_dict = {
        'missing': missing,
        'duplicates': duplicates,
        'negative_prices': negative,
        'negative_data': negative_data,
        'close_open_anomalies': anomalies,
        'clipped_open': clipped_open,
    }
    return checks_dict

def fill_data(data, method='ffill'):
    """
        Fill missing values in the dataset using specified fill method. Used in cleaning step.
        Parameters:
            data (pd.DataFrame): DataFrame containing stock data with missing values to be filled.
            method (str): Method to fill missing values. Options include 'ffill' for forward fill,
                          'bfill' for backward fill. Defaults to 'ffill' if called without correct argument
                           for method.
        Returns:
            filled_data (pd.DataFrame): DataFrame with missing values filled according to specified method.
    """
    if method == 'ffill':
        filled_data = data.ffill()
    elif method == 'bfill':
        filled_data = data.bfill()
    else:
        print(f"Invalid fill method specified. Allowed options are 'ffill' or 'bfill'. Defaulting to 'ffill'.")
        filled_data = data.ffill()

    return filled_data

def clean_data(data, checks_dict, fill_method='ffill'):
    """
   Step 3: Data Cleaning + Imputation
   Handle missing, outlier, negative and duplicated values and anomalies.
   Imputes any removed values using specified fill_method to ensure continuity of data.
  
   Parameters:
       data (pd.DataFrame): Raw stock data with potential missing values.
       fill_method (str): Method to fill missing values ('ffill', 'bfill', 'mean' options).

    Returns:
        cleaned_data (pd.DataFrame): Cleaned DataFrame with missing values and duplicates handled.
    """

    if not checks_dict:
        checks_dict = validate_data(data)
    
    dict_requirements = ['missing', 
                        'duplicates', 
                        'negative_prices', 
                        'negative_data', 
                        'close_open_anomalies', 
                        'clipped_open'
                        ]

    if not all(key in checks_dict for key in dict_requirements):
        print(f"Checks dictionary is missing required keys: {dict_requirements}. Please ensure validate_data function is run and returns all required checks.")
        sys.exit(1)

    #handles close-open anomalies by clipping open prices to threshold if specified:
    if checks_dict['close_open_anomalies']:
        print("Handling close-open anomalies by clipping open prices.")
        clipped = checks_dict['clipped_open'].dropna()
        data['Open'] = clipped

    #Handle negative price rows:
    if checks_dict['negative_prices']:
        print("Handling negative price rows by removing them from dataset.")
        data = data[~data.index.isin(checks_dict['negative_data'].index)]

    #drop duplicates if needed and specified:
    if checks_dict['duplicates']:
        data = data[~data.index.duplicated(keep='first')]
        print("Duplicate rows dropped.")

    #Fill any missing values using specified method:
    if checks_dict['missing']:
        data = fill_data(data, method=fill_method)
    
    #handle any final missing values post cleaning if any by filling them:
    if data.isna().any().any():
        data = data.ffill().bfill() # if any missing values remain after cleaning then fill them using forward and backward fill to ensure no missing values remain.

    return data['Close'] #only returns adjusted close price data

def plot_single_stock(data, symbol):
    """
    Utility function to plot the closing price of a single stock over time.
    Parameters:
        data (pd.DataFrame): DataFrame containing stock data with 'Close' price column.
        symbol (str): Stock ticker symbol to be plotted.
    Returns:
        None: Displays the plot of closing price for the specified stock.
    """
    if symbol not in data.columns:
        print(f"Symbol {symbol} not found in data. Available symbols are: {data.columns.tolist()}")
        return None

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[symbol], label=f'{symbol} Close Price', linewidth=2)
    plt.title(f'{symbol} Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_stocks(data):
    """
    Utility function to plot the closing price of multiple stocks. Creates one plot per stock.
    Parameters:
        data (pd.DataFrame): DataFrame containing stock data with 'Close' price columns for multiple stocks.
    Returns:
        None: Displays individual plots for each stock.
    """
    plt.figure(figsize=(12, 6))
    for symbol in cf.SYMBOLS:
        if symbol in data.columns:
            plt.plot(data.index, data[symbol], label=f'{symbol} Close Price', linewidth=2)
        else:
            print(f"Symbol {symbol} not found in data. Available symbols are: {data.columns.tolist()}")
    plt.title('Closing Price of Stocks Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

def save_plots(data, plot, filename='stock_price_plots.png'):
    """
    Utility function to save the plot of closing price of multiple stocks. Creates one plot per stock and saves it to specified location.
    Parameters:
        data (pd.DataFrame): DataFrame containing stock data with 'Close' price columns for multiple stocks.
        plot (plt.Figure): The plot object to save.
        filename (str): Name of the file to save the plot to.
    Returns:
        None: Saves the plot of closing price for the specified stocks to a file.
    """
    pass #tbd in a bit



def save_data(data, filename='cleaned_stock_data.csv'):
    """
    Step 4: Save Data
    Save the cleaned and processed data to a specified location for future use in analysis and modeling.
    Parameters:
        data (pd.DataFrame): Cleaned stock data to be saved.
        filename (str): Name of the file to save the data to.
    Returns:
        None: Saves the cleaned data to a specified file path and prints file path for reference.
    """
    cf.ensure_directories() # ensure that necessary directories exist before saving data.
    data.to_csv(cf.PROCESSED_DATA / filename)
    print(f"Cleaned data for stocks {cf.SYMBOLS} saved to {cf.PROCESSED_DATA / filename}")

def main():
   stock_data = fetch_stock_data(cf.SYMBOLS, cf.START_DATE, cf.END_DATE, cf.INTERVAL)
   checks = validate_data(stock_data)
   cleaned_data = clean_data(stock_data, checks_dict=checks)
   save_data(cleaned_data)
   print(cleaned_data.head())

if __name__ == "__main__":
    main()