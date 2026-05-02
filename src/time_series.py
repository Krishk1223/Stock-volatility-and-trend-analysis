import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import config as cf
from src.data_pipeline import plot_single_stock, plot_stocks
from src.data_pipeline import load_data, save_data, save_plots
from scipy.stats import norm, probplot as qqplot
from statsmodels.graphics.tsaplots import plot_acf as acf
from statsmodels.graphics.tsaplots import plot_pacf as pacf
import seaborn as sns

def log_returns(data, symbol, drop=False):
    """
    Calculate log returns for given stock data (and optionally drops NaN values).
    Parameters:
        data (pd.DataFrame): DataFrame containing stock data with 'Close' price column.
        symbol (str): The stock symbol for which to calculate log returns.
        drop (bool) : whether to drop NaN values resulting from log return calculation. Defaults to false
    Returns:
        log_returns (pd.DataFrame): DataFrame containing log returns of the stock prices.
    """
    if symbol not in data.columns:
        raise ValueError(f"Symbol {symbol} not found in data columns. Available columns: {data.columns}")

    log_returns = np.log(data[symbol] / data[symbol].shift(1))
    if drop:
        log_returns = log_returns.dropna()
    return log_returns

def features(data):
    """
    Add log returns as a feature as well as squared log returns. This will be where we create new features from the raw data, 
    such as log returns.
    Parameters:
        data (pd.DataFrame): Cleaned stock data to be used for feature engineering.
    Returns:
        features (pd.DataFrame): DataFrame with new features added.
    """
    features = data.copy()
    for symbol in cf.SYMBOLS:
        features[f'{symbol}_Log_Returns'] = log_returns(data, symbol, drop=False)
        features[f'{symbol}_Squared_Log_Returns'] = features[f'{symbol}_Log_Returns'] ** 2

    return features.dropna()

def plot_single_log_return(data, symbol):
    """
    Plot log returns for a single stock.
    Parameters:
        data (pd.DataFrame): DataFrame containing log returns of the stock prices.
        symbol (str): The stock symbol for which to plot log returns.
    Returns:
        None: Displays a plot of log returns over time for the specified stock.
    """
    column_name = f'{symbol}_Log_Returns'
    if column_name not in data.columns:
        raise ValueError(f"Log returns for symbol {symbol} not found in data columns. Available columns: {data.columns}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[column_name], label=f'{symbol} Log Returns')
    plt.title(f'{symbol} Log Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Log Return')
    plt.legend()
    plt.grid()
    plt.show()

def plot_all_log_returns(data):
    """
    Plot log returns for all stocks in the portfolio.
    Parameters:
        data (pd.DataFrame): DataFrame containing log returns of the stock prices.
    Returns:
        None: Displays a plot of log returns over time for all stocks in the portfolio.
    """
    plt.figure(figsize=(12, 6))
    for symbol in cf.SYMBOLS:
        column_name = f'{symbol}_Log_Returns'
        if column_name not in data.columns:
            print(f"Log returns for symbol {symbol} not found in data columns. Skipping plot for this symbol.")
            continue
        plt.plot(data.index, data[column_name], label=f'{symbol} Log Returns')
    
    plt.title('Log Returns Over Time for All Stocks')
    plt.xlabel('Date')
    plt.ylabel('Log Return')
    plt.legend()
    plt.grid()
    plt.show()

def descriptive_statistics(data,  kurtosis_threshold=cf.KURTOSIS_THRESHOLD):
    """
    Calculate financial statistics for log returns. Calculations are as follows: mean, std, skewness, kurtosis

    Parameters:
        data (pd.DataFrame): DataFrame containing log returns of the stock prices.
        kurtosis_threshold (float): Threshold for excess kurtosis to flag potential outliers.
    Returns:
        stats (pd.DataFrame): DataFrame containing descriptive statistics of the log returns.
    """
    for symbol in cf.SYMBOLS:
        if f'{symbol}_Log_Returns' not in data.columns:
            raise ValueError(f"Log returns for symbol {symbol} not found in data columns. Available columns: {data.columns}")

        column_name = f'{symbol}_Log_Returns'
        data[f'{symbol}_rolling_mean'] = data[column_name].rolling(window=cf.ROLLING_WINDOW).mean()
        data[f'{symbol}_rolling_std'] = data[column_name].rolling(window=cf.ROLLING_WINDOW).std()
        data[f'{symbol}_skewness'] = data[column_name].rolling(window=cf.ROLLING_WINDOW).skew()
        data[f'{symbol}_kurtosis'] = data[column_name].rolling(window=cf.ROLLING_WINDOW).kurt()
        data[f'{symbol}_annualised_volatility'] = data[f'{symbol}_rolling_std'] * np.sqrt(cf.ANNUALISED_WINDOW)
        data[f'{symbol}_potential_outlier'] = data[f'{symbol}_kurtosis'] > kurtosis_threshold
    
    return data

def plot_rolling_statistics(data, symbol, ax=None):
    """
    Plot rolling mean and std dev of log returns.
    Parameters:
        data (pd.DataFrame): DataFrame containing log returns of the stock prices.
        symbol (str): The stock symbol for which to plot rolling statistics.
    Returns:
        fig (plt.Figure): A matplotlib Figure object containing the plot showing the log returns along with rolling mean and rolling std dev for 
              the specified stock.
    """
    symbol_mean_col = f'{symbol}_rolling_mean'
    symbol_std_col = f'{symbol}_rolling_std'
    symbol_log_return_col = f'{symbol}_Log_Returns'
    if symbol_mean_col not in data.columns or symbol_std_col not in data.columns:
        raise ValueError(f"Rolling mean or std dev for symbol {symbol} not found in data columns. Available columns: {data.columns}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    ax.plot(data.index, data[symbol_mean_col], label='Rolling Mean', linewidth=2)
    ax.plot(data.index, data[symbol_log_return_col], label='Log Returns', alpha=0.5)
    ax.fill_between(data.index, 
                     data[symbol_mean_col] - data[symbol_std_col],
                     data[symbol_mean_col] + data[symbol_std_col],
                     alpha=0.4, label='±1 Std Dev')
    ax.set_title(f'{symbol} Log Returns (Rolling Statistics)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Log Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return fig

def plot_annualised_volatility(data, symbol, ax=None):
    """
    Plot annualised volatility of log returns.
    Parameters:
        data (pd.DataFrame): DataFrame containing log returns of the stock prices.
        symbol (str): The stock symbol for which to plot annualised volatility.
    Returns:
        fig (plt.Figure): A matplotlib Figure object containing the plot showing the annualised volatility of log returns for the specified stock.
    """

    symbol_vol_col = f'{symbol}_annualised_volatility'
    if symbol_vol_col not in data.columns:
        raise ValueError(f"Annualised volatility for symbol {symbol} not found in data columns. Available columns: {data.columns}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    ax.plot(data.index, 100*data[symbol_vol_col], label='Annualised Volatility', color='orange')
    ax.set_title(f'{symbol} Annualised Volatility Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Annualised Volatility (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return fig

def plot_returns_distribution(data, symbol, bin_count=50, ax=None, range_theta=4):
    """
    Plot histogram of log returns.
    Parameters:
        data (pd.DataFrame): DataFrame containing log returns of the stock prices.
        symbol (str): The stock symbol for which to plot log returns distribution.
        bin_count (int): Number of bins for the histogram. Defaults to 50.
        ax (matplotlib.axes.Axes, optional): Matplotlib Axes object to plot on. 
                                             If None, a new figure and axes will be created. Defaults to None.
        range_theta (INT): Range in terms of multiples of standard deviations for x-axis limits. Defaults to 4.
    Returns:
        fig (plt.Figure): A matplotlib Figure object containing the histogram and normal distribution overlay.
    """
    # for normal distribution
    symbol_log_return_col = f'{symbol}_Log_Returns'
    global_log_ret_mean = data[symbol_log_return_col].mean()
    global_log_ret_std = data[symbol_log_return_col].std()
    norm_range = np.linspace(global_log_ret_mean - range_theta*global_log_ret_std,
                             global_log_ret_mean + range_theta*global_log_ret_std, 1000)

    column_name = f'{symbol}_Log_Returns'
    if column_name not in data.columns:
        raise ValueError(f"Log returns for symbol {symbol} not found in data columns. Available columns: {data.columns}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()
    
    ax.hist(data[column_name].dropna(), bins=bin_count, edgecolor='black', alpha=0.7, density=True)
    ax.plot(norm_range, norm.pdf(norm_range, global_log_ret_mean, global_log_ret_std), 'r-', linewidth=2, label='Normal Distribution')
    ax.set_title(f'{symbol} Log Returns Distribution')
    ax.set_xlabel('Log Returns')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig

def plot_qq(data, symbol, ax=None):
    """
    Plot Q-Q plot of log returns to assess normal nature and see tail behaviour + size.
    Parameters:
        data (pd.DataFrame): DataFrame containing log returns of the stock prices.
        symbol (str): The stock symbol for which to plot Q-Q plot.
        ax (matplotlib.axes.Axes, optional): Matplotlib Axes object to plot on. 
                                             If None, a new figure and axes will be created. Defaults to None.
    Returns:
        fig (plt.Figure): A matplotlib Figure object containing the Q-Q plot of log returns
    """
    column_name = f'{symbol}_Log_Returns'

    if column_name not in data.columns:
        raise ValueError(f"Log returns for symbol {symbol} not found in data columns. Available columns: {data.columns}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    qqplot(data[column_name].dropna(), dist=norm, plot=ax)
    ax.set_title(f'{symbol} Q-Q Plot')
    ax.set_xlabel('Theoretical Quantiles (Normal Distribution)')
    ax.set_ylabel('Sample Quantiles (Stock Log Returns)')
    fig.tight_layout()

    return fig

def plot_acf(data, symbol, squared=False, ax=None):
    """
    Plot autocorrelation function of specified data (log returns for ARIMA diagnostics 
    and squared log returns for GARCH diagnostics). Uses 10 * log(length of data) as number of lags to plot.
    Parameters:
        data (pd.DataFrame): DataFrame containing log returns of the stock prices.
        symbol (str): The stock symbol for which to plot autocorrelation function.
        squared (bool): Whether to plot squared log returns for GARCH diagnostics. Defaults to False.
        ax (matplotlib.axes.Axes, optional): Matplotlib Axes object to plot on. 
                                             If None, a new figure and axes will be created. Defaults to None.
    Returns:
        fig (plt.Figure): A matplotlib Figure object containing the autocorrelation function plot.
    """
    if squared:
        type = 'Squared Log Returns'
        column_name = f'{symbol}_Squared_Log_Returns'
    else:
        type = 'Log Returns'
        column_name = f'{symbol}_Log_Returns'

    if column_name not in data.columns:
        raise ValueError(f"{column_name} for symbol {symbol} not found in data columns. Available columns: {data.columns}")

    lag_count = 10 * int(np.log(len(data[column_name].dropna())))

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    fig = acf(data[column_name].dropna(), ax=ax, lags=lag_count, title=f'{symbol} {type} Autocorrelation')
    ax.set_xlabel('Lags')
    ax.set_ylabel('Autocorrelation')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig

def plot_pacf(data, symbol, squared=True, ax=None):
    """
    Plot partial autocorrelation function of specified data (squared log returns for GARCH diagnostics).
    Uses 10 * log(length of data) as number of lags to plot.
    Parameters:
        data (pd.DataFrame): DataFrame containing log returns of the stock prices.
        symbol (str): The stock symbol for which to plot partial autocorrelation function.
        squared (bool): Whether to plot squared log returns for GARCH diagnostics. Defaults to True.
        ax (matplotlib.axes.Axes, optional): Matplotlib Axes object to plot on. 
                                             If None, a new figure and axes will be created. Defaults to None.
    Returns:
        fig (plt.Figure): A matplotlib Figure object containing the partial autocorrelation function plot.
    """
    if squared:
        type = 'Squared Log Returns'
        column_name = f'{symbol}_Squared_Log_Returns'
    else:
        type = 'Log Returns'
        column_name = f'{symbol}_Log_Returns'

    if column_name not in data.columns:
        raise ValueError(f"{column_name} for symbol {symbol} not found in data columns. Available columns: {data.columns}")

    lag_count = 10 * int(np.log(len(data[column_name].dropna())))
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()
    
    fig = pacf(data[column_name].dropna(), ax=ax, lags=lag_count, title=f'{symbol} {type} Partial Autocorrelation')
    ax.set_xlabel('Lags')
    ax.set_ylabel('Partial Autocorrelation')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig

def stock_show_plots(data, symbol):
    """
    Show all relevant plots for a single stock.
    Parameters:
        data (pd.DataFrame): DataFrame containing log returns of the stock prices and related statistics.
        symbol (str): The stock symbol for which to show plots.
    Returns:
        figures (plt.Figure): A matplotlib Figure objects containing the plots for the specified stock.
                                Plots included:
                                - Log returns over time with rolling mean and std dev
                                - Annualised volatility over time
                                - Log returns distribution
                                - histogram of log returns with normal distribution overlay
                                - autocorrelation function
                                - partial autocorrelation function
    """
    figure, axes = plt.subplots(2, 2, figsize=(15, 8))
    plot_rolling_statistics(data, symbol, ax=axes[0, 0])
    plot_annualised_volatility(data, symbol, ax=axes[0, 1])
    plot_returns_distribution(data, symbol, ax=axes[1, 0])
    plot_qq(data, symbol, ax=axes[1, 1])

    return figure

def stock_show_acf_pacf(data, symbol):
    """
    Show ACF and PACF plots for log returns and squared log returns for a single stock.
    Parameters:
        data (pd.DataFrame): DataFrame containing log returns of the stock prices and related statistics.
        symbol (str): The stock symbol for which to show ACF and PACF plots.
    Returns:
        figures (plt.Figure): A matplotlib Figure objects containing the ACF and PACF plots for the specified stock.
                                Plots included:
                                - autocorrelation function for log returns
                                - autocorrelation function for squared log returns
                                - partial autocorrelation function for log returns
                                - partial autocorrelation function for squared log returns
    """
    figure, axes = plt.subplots(2, 2, figsize=(15, 8))
    plot_acf(data, symbol, squared=False, ax=axes[0, 0])
    plot_acf(data, symbol, squared=True, ax=axes[0, 1])
    plot_pacf(data, symbol, squared=False, ax=axes[1, 0])
    plot_pacf(data, symbol, squared=True, ax=axes[1, 1])

    return figure

def correlation_matrix(data):
    """
    Calculate correlation matrix of log returns for all stocks in the portfolio.
    Parameters:
        data (pd.DataFrame): DataFrame containing log returns of the stock prices.
    Returns:
        fig (plt.Figure): Matplotlib Figure object containing the correlation matrix heatmap.
    """
    data = data.copy()
    log_return_cols = [f'{symbol}_Log_Returns' for symbol in cf.SYMBOLS if f'{symbol}_Log_Returns' in data.columns]
    if not log_return_cols:
        raise ValueError("No log return columns found in data. Ensure that log returns have been calculated and added to the DataFrame.")
    
    corr_matrix = data[log_return_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    ax.set_title('Correlation Matrix of Log Returns')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    
    return fig

def covariance_matrix(data):
    """
    Calculate covariance matrix of log returns for all stocks in the portfolio.
    Parameters:
        data (pd.DataFrame): DataFrame containing log returns of the stock prices.
    Returns:
        fig (plt.Figure): Matplotlib Figure object containing the covariance matrix heatmap.
    """
    data = data.copy()
    log_return_cols = [f'{symbol}_Log_Returns' for symbol in cf.SYMBOLS if f'{symbol}_Log_Returns' in data.columns]
    if not log_return_cols:
        raise ValueError("No log return columns found in data. Ensure that log returns have been calculated and added to the DataFrame.")
    
    cov_matrix = data[log_return_cols].cov()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(cov_matrix, annot=True, cmap='coolwarm')
    ax.set_title('Covariance Matrix of Log Returns')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    
    return fig

def returns_data(data):
    """
    Extract log returns data for all stocks in the portfolio.
    Parameters:
        data (pd.DataFrame): DataFrame containing log returns of the stock prices.
    Returns:
        returns_df (pd.DataFrame): DataFrame containing log returns of all stocks in the portfolio.
    """
    log_return_cols = [f'{symbol}_Log_Returns' for symbol in cf.SYMBOLS if f'{symbol}_Log_Returns' in data.columns]
    if not log_return_cols:
        raise ValueError("No log return columns found in data. Ensure that log returns have been calculated and added to the DataFrame.")
    
    returns_df = data[log_return_cols].dropna()
    return returns_df

def time_series_pipeline():
    # Load cleaned data:
    cleaned_data = load_data(filename='cleaned_portfolio_data.csv')
    # Feature engineering:
    features_data = features(cleaned_data)
    features_df = descriptive_statistics(features_data)
    time_series_plots = []
    acf_pacf_plots = []
    for symbol in cf.SYMBOLS:
        time_series_plots.append(stock_show_plots(features_df, symbol))
        acf_pacf_plots.append(stock_show_acf_pacf(features_df, symbol))
    returns_df = returns_data(features_df)
    corr_fig = correlation_matrix(features_df)
    cov_fig = covariance_matrix(features_df)

    #Save all plots to output directory:
    for i, symbol in enumerate(cf.SYMBOLS):
        save_plots(time_series_plots[i], directory=cf.TIME_SERIES_DIR, filename=f'{symbol}_time_series_plots.png')
        save_plots(acf_pacf_plots[i], directory=cf.TIME_SERIES_DIR, filename=f'{symbol}_acf_pacf_plots.png')
    
    save_plots(corr_fig, directory=cf.TIME_SERIES_DIR, filename='correlation_matrix.png')
    save_plots(cov_fig, directory=cf.TIME_SERIES_DIR, filename='covariance_matrix.png')

    # save returns data to processed data directory for use in modelling:
    save_data(returns_df, filename='log_returns_data.csv')

    print("Time series analysis pipeline completed successfully.")

    return None


def main():
    cf.ensure_directories()
    time_series_pipeline()

if __name__ == "__main__":
    main()
