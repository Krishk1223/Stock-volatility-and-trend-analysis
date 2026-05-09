import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import config as cf
from src.data_pipeline import load_data, save_plots
from pmdarima import auto_arima
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Data splitting:
def train_test_split(returns_data, test_size=cf.FORECAST_STEPS):
    """
    Splits log returns data into training and test sets based on specified test size 
    (number of steps to forecast).
    Parameters:
        returns_data (pd.DataFrame): DataFrame containing log returns data for stocks.
        test_size (int): Number of time steps to include in the test set (default is config.FORECAST_STEPS).
    Returns:
        train_data (pd.DataFrame): DataFrame containing training set log returns data.
        test_data (pd.DataFrame): DataFrame containing test set log returns data.
    """
    split_index = int(len(returns_data) - test_size)
    train_data = returns_data.iloc[:split_index]
    test_data = returns_data.iloc[split_index:]

    return train_data, test_data

# ARIMA model functions:
def fit_arima(train_data, stock_symbol):
    """
    Fits an AutoRegressive Integrated Moving Average (ARIMA) model to log returns data for specified stock 
    log returns data. Uses auto_arima to determine order of ARIMA model.
    Parameters:
        train_data (pd.DataFrame): Training set containing log returns data for stocks.
        stock_symbol (str): Symbol of stock symbol for which to fit ARIMA model.
    Returns:
        model: Fitted ARIMA model object.
    """
    column = f'{stock_symbol}_Log_Returns'
    if column not in train_data.columns:
        raise ValueError(f"Column '{column}' not found in train_data.")
    stock_returns = train_data[column].dropna()
    model = auto_arima(stock_returns, 
                       seasonal=False, 
                       stepwise=True,
                       trace=cf.ARIMA_TRACE,
                       error_action='ignore',
                       suppress_warnings=True,
                       max_p=cf.ARIMA_MAX_P,
                       max_d=cf.ARIMA_MAX_D,
                       max_q=cf.ARIMA_MAX_Q,
                       information_criterion=cf.ARIMA_CRITERION
                       )

    return model

def forecast_arima(model, steps=cf.FORECAST_STEPS):
    """
    Generate forecasts from fitted ARIMA model for specified number of steps and confidence level.
    Parameters:
        model: Fitted ARIMA model object.
        steps (int): Number of future time steps to forecast (default is config.FORECAST_STEPS).
    Returns:
        forecast (pd.Series): Forecasted values for specified number of steps.
    """
    forecast_result = model.predict(n_periods=steps)
    return forecast_result

def evaluate_arima(model, forecast, stock_symbol, test_data):
    """
    Evaluates ARIMA model forecast against actual log returns in test set using MSE, MAE, RMSE 
    and R-squared metrics.
    Parameters:
        model: Fitted ARIMA model object.
        forecast (pd.Series): Forecasted log returns from ARIMA model.
        stock_symbol (str): Symbol of the stock for which to evaluate the forecast.
        test_data (pd.DataFrame): Test set containing log returns data for stocks.
    Returns:
        metrics (dict): Dictionary containing evaluation metrics (MSE, MAE, RMSE, R-squared) for the 
        ARIMA forecast.
    """
    column = f'{stock_symbol}_Log_Returns'
    if column not in test_data.columns:
        raise ValueError(f"Column '{column}' not found in test_data.")
    actual = test_data[column].dropna().values
    mse = mean_squared_error(actual, forecast)
    mae = mean_absolute_error(actual, forecast)
    r2 = r2_score(actual, forecast)
    
    metrics = {'MSE': mse, 'RMSE': np.sqrt(mse), 'MAE': mae, 'R-squared': r2}
    parameters = {
        'order': model.order
        }

    return metrics, parameters

def convert_forecast_to_df(forecast, returns_data, stock_symbol):
    """
    Converts forecasted values from ARIMA model into a DataFrame with appropriate date index and column name.
    Parameters:
        forecast (pd.Series): Series containing forecasted log returns from ARIMA model.
        returns_data (pd.DataFrame): DataFrame containing log returns data for stocks.
        stock_symbol (str): Symbol of the stock for which to convert the forecast.
    Returns:
        forecast_df (pd.DataFrame): DataFrame containing forecasted log returns with date index and column name.
    """
    start_index = len(returns_data) - cf.FORECAST_STEPS
    forecast_index = pd.date_range(start=returns_data.index[start_index] + pd.Timedelta(days=1), periods=len(forecast), freq='B') # business day frequency for stock data
    forecast_df = pd.DataFrame(forecast.values, index=forecast_index, columns=[f'{stock_symbol}_ARIMA_Forecast'])
    
    return forecast_df

def plot_arima_forecast(returns_data, forecast_df, stock_symbol, ax=None):
    """
    Plots actual log returns vs ARIMA forecasted log returns for specified stock symbol.
    Parameters:
        returns_data (pd.DataFrame): DataFrame containing log returns data for stocks.
        forecast_df (pd.DataFrame): DataFrame containing forecasted log returns from ARIMA model.
        stock_symbol (str): Symbol of the stock for which to plot the forecast.
        ax (matplotlib.axes.Axes, optional): Matplotlib Axes object to plot on. If None, creates new figure and axes.
    Returns:
        fig (matplotlib.figure.Figure): Figure object containing the plot.
    """
    column = f'{stock_symbol}_Log_Returns'
    if column not in returns_data.columns:
        raise ValueError(f"Column '{column}' not found in returns_data.")
    actual = returns_data.iloc[-cf.FORECAST_STEPS:][column].dropna() # get actual log returns for the test period (last cf.FORECAST_STEPS rows)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    ax.plot(actual.index, actual.values, label='Actual Log Returns', color='blue', alpha=0.3)
    ax.plot(forecast_df.index, forecast_df.iloc[:, 0], label='ARIMA Forecast', color='orange', linewidth=2)
    ax.set_title(f'ARIMA Forecast vs Actual Log Returns for {stock_symbol}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Log Returns')
    ax.legend()
    fig.tight_layout()

    return fig


# GARCH model functions:
def fit_garch(train_data, test_data, stock_symbol, distribution=cf.GARCH_DIST, scale=cf.GARCH_SCALING, type=cf.ARCH_TYPE):
    """
    Fits a GARCH model to log returns data for specified stock symbol using a rolling window approach,
    generating forecasts for each time step in the test set. Returns a dictionary containing forecasted
    volatilities, actual test returns, and final fitted model results for parameter extraction.
    Parameters:
        train_data (pd.DataFrame): Training set containing log returns data for stocks.
        test_data (pd.DataFrame): Test set containing log returns data for stocks.
        stock_symbol (str): Symbol of the stock for which to fit the GARCH model.
        distribution (str): Distribution for error terms in GARCH model ('Normal' or 'StudentsT').
        scale (float): Scaling factor for returns when fitting GARCH model to help optimiser convergence.
        type (str): Type of ARCH model to fit (e.g., 'GARCH', 'EGARCH').
    Returns:
        rolling_results (dict): Dictionary containing forecasted volatilities, actual test returns, and final fitted model
                                 results for parameter extraction. (Or None)
    """
    if distribution not in ['Normal', 'StudentsT']:
        raise ValueError("Invalid distribution specified. Use 'Normal' for normal or 'StudentsT' for Student's t.")

    column = f'{stock_symbol}_Log_Returns'
    if column not in train_data.columns:
        raise ValueError(f"Column '{column}' not found in train_data.")

    all_returns = train_data[column].dropna().copy()
    test_returns = test_data[column].dropna().copy()
    
    forecasted_volatilities = []
    final_results = None
    
    for i in range(len(test_returns)):
        try:
            current_train = pd.concat([all_returns, test_returns.iloc[:i]])
            
            model = arch_model(
                current_train * scale,
                vol=cf.ARCH_TYPE,
                mean=cf.GARCH_MEAN,
                p=cf.GARCH_P,
                q=cf.GARCH_Q,
                dist=distribution
            )
            results = model.fit(disp='off')
            final_results = results
            
            forecast = results.forecast(horizon=1)
            forecasted_var = forecast.variance.iloc[-1].values[0] / scale**2
            forecasted_vol = np.sqrt(forecasted_var)
            forecasted_volatilities.append(forecasted_vol)
            
        except Exception as e:
            if forecasted_volatilities:
                forecasted_volatilities.append(forecasted_volatilities[-1])
            else:
                forecasted_volatilities.append(np.nan)
    
    rolling_results = {
        'forecasted_volatilities': np.array(forecasted_volatilities),
        'test_returns': test_returns,
        'stock_symbol': stock_symbol,
        'final_results': final_results,
        'scale': scale
    }
    
    return rolling_results, None

def extract_garch_params(rolling_results, forecast=None, scale=cf.GARCH_SCALING):
    """
    Extracts key parameters and metrics from GARCH model fitting results, including forecasted volatilities,
    realised volatility, mean forecasted volatility, mean and std of realised volatility, and GARCH
    parameters (alpha, beta, omega, persistence) if available.
    Parameters:
        rolling_results (dict): Dictionary containing forecasted volatilities, actual test returns,
                                and final fitted model results for parameter extraction.
        forecast (pd.Series, optional): Series containing forecasted log returns from ARIMA model, 
                                        used for context in GARCH evaluation.
        scale (float): Scaling factor used for returns when fitting GARCH model, 
                        used to rescale parameters if needed.

    Returns:
        params_dict (dict): Dictionary containing extracted parameters and metrics from GARCH model 
                            fitting results.
    """
    forecasted_volatilities = rolling_results['forecasted_volatilities']
    test_returns = rolling_results['test_returns']
    final_results = rolling_results.get('final_results')
    garch_scale = rolling_results.get('scale', scale)
    
    realised_variance = test_returns.values ** 2
    realised_volatility = np.sqrt(realised_variance)
    
    params_dict = {
        'forecasted_volatilities': forecasted_volatilities,
        'realised_volatility': realised_volatility,
        'mean_forecasted_vol': forecasted_volatilities.mean(),
        'mean_realised_vol': realised_volatility.mean(),
        'std_realised_vol': realised_volatility.std(),
        'test_returns': test_returns
    }
    
    if final_results is not None:
        try:
            if cf.GARCH_MEAN == 'Constant':
                params_dict['mu'] = float(final_results.params['mu'] / garch_scale)
            params_dict['alpha'] = float(final_results.params['alpha[1]'])
            params_dict['beta'] = float(final_results.params['beta[1]'])
            params_dict['omega'] = float(final_results.params['omega'] / garch_scale ** 2)
            params_dict['persistence'] = float(final_results.params['alpha[1]'] + final_results.params['beta[1]'])
            params_dict['dof'] = float(final_results.params['nu']) if cf.GARCH_DIST == 'StudentsT' else None
            params_dict['residuals'] = final_results.resid/garch_scale
            params_dict['last_epsilon'] = float(params_dict['residuals'].iloc[-1])
            params_dict['last_sigma'] = float(final_results.conditional_volatility.iloc[-1]/garch_scale)
        except Exception as e:
            print(f"Warning: Could not extract some GARCH parameters: {e}")
    
    return params_dict

def plot_garch_forecast_volatility(train_data, test_data, results_dict, stock_symbol, ax=None):
    """
    Plots forecasted volatility from rolling GARCH model against realised volatility for specified stock symbol.
    Parameters:
        train_data (pd.DataFrame): Training set containing log returns data for stocks.
        test_data (pd.DataFrame): Test set containing log returns data for stocks.
        results_dict (dict): Dictionary containing forecasted volatilities, actual test returns, and
                                final fitted model results for parameter extraction.
        stock_symbol (str): Symbol of the stock for which to plot the GARCH forecast.
        ax (matplotlib.axes.Axes, optional): Matplotlib Axes object to plot on. if none then creates new figure.
    Returns:
        fig (matplotlib.figure.Figure): Figure object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()
    
    test_returns = results_dict['test_returns']
    forecasted_vol = results_dict['forecasted_volatilities']
    realised_vol = results_dict['realised_volatility']
    
    ax.plot(test_returns.index, forecasted_vol,
            label='Rolling GARCH Forecast', color='orange', linewidth=2)
    ax.plot(test_returns.index, realised_vol,
            label='Realised Volatility', color='blue', alpha=0.6)
    
    ax.set_title(f'Rolling GARCH({cf.GARCH_P},{cf.GARCH_Q}) Volatility Forecast for {stock_symbol}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return fig

def evaluate_garch(results_dict, test_data, stock_symbol):
    """
    Evaluates GARCH model forecasted volatilities against realised volatility using MSE, MAE, RMSE 
    and QLIKE metrics.
    Parameters:
        results_dict (dict): Dictionary containing forecasted volatilities, actual test returns, and
                                final fitted model results for parameter extraction.
        test_data (pd.DataFrame): Test set containing log returns data for stocks.
        stock_symbol (str): Symbol of the stock for which to evaluate the GARCH forecast.
    Returns:
        metrics (dict): Dictionary containing evaluation metrics (MSE, MAE, RMSE and QLIKE) for GARCH forecast.
        parameters (dict): Dictionary containing GARCH model parameters (alpha, beta, omega, persistence) if available.
    """
    forecasted_vol = results_dict['forecasted_volatilities']
    realised_vol = results_dict['realised_volatility']
    
    forecasted_var = forecasted_vol ** 2
    realised_var = realised_vol ** 2
    
    mse = mean_squared_error(realised_var, forecasted_var)
    mae = mean_absolute_error(realised_vol, forecasted_vol)
    rmse = np.sqrt(mse)
    qlike = np.mean(np.log(forecasted_var) + realised_var / forecasted_var)
    
    mean_realized_vol = results_dict['mean_realised_vol']
    
    metrics = {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'QLIKE': float(qlike),
        'Mean_Forecasted_Volatility': float(forecasted_vol.mean()),
        'Mean_Realised_Volatility': float(mean_realized_vol),
        'Std_Realised_Volatility': float(results_dict['std_realised_vol']),
    }
    
    parameters = {
        'mean_forecasted_vol': results_dict.get('mean_forecasted_vol'),
        'mean_realised_vol': results_dict.get('mean_realised_vol'),
        'std_realised_vol': results_dict.get('std_realised_vol'),
    }
    if 'mu' in results_dict:
        parameters['mu'] = float(results_dict['mu'])
    if 'alpha' in results_dict:
        parameters['alpha'] = float(results_dict['alpha'])
    if 'beta' in results_dict:
        parameters['beta'] = float(results_dict['beta'])
    if 'omega' in results_dict:
        parameters['omega'] = float(results_dict['omega'])
    if 'persistence' in results_dict:
        parameters['persistence'] = float(results_dict['persistence'])
    if 'dof' in results_dict and results_dict['dof'] is not None:
        parameters['dof'] = float(results_dict['dof'])
    if 'last_epsilon' in results_dict:
        parameters['last_epsilon'] = float(results_dict['last_epsilon'])
    if 'last_sigma' in results_dict:
        parameters['last_sigma'] = float(results_dict['last_sigma'])
    
    return metrics, parameters

# saving parameters:
def save_parameters_pickle(arima_params, stock_symbol, garch_params, output_dir=cf.MODEL_PARAMATERS):
    """
    Saves ARIMA and GARCH model parameters for specified stock symbol to pickle files in output directory.
    Parameters:
        arima_params (dict): Dictionary containing ARIMA model parameters for the specified stock symbol.
        stock_symbol (str): Symbol of the stock for which to save model parameters.
        garch_params (dict): Dictionary containing GARCH model parameters for the specified stock symbol. 
        output_dir (str or Path, optional): Directory where model parameter pickle files should be saved. 
                                            If None, defaults to config.MODEL_PARAMATERS.
    Returns:
        None
    """
    output_dir = Path(output_dir)
    
    arima_file = output_dir / f'{stock_symbol}_arima_parameters.pkl'
    with open(arima_file, 'wb') as f:
        pickle.dump(arima_params, f)
    
    garch_file = output_dir / f'{stock_symbol}_{cf.ARCH_TYPE}_parameters.pkl'
    with open(garch_file, 'wb') as f:
        pickle.dump(garch_params, f)

    return None

# loading parameters:
def load_arima_parameters(stock_symbol, input_dir=None):
    """
    Loads ARIMA model parameters for specified stock symbol from pickle file in input directory.
    Parameters:
        stock_symbol (str): Symbol of the stock for which to load ARIMA parameters.
        input_dir (str or Path, optional): Directory where ARIMA parameter pickle files are stored.
                                             If None, defaults to config.MODEL_PARAMATERS.
    Returns:
        parameters (dict): Dictionary containing ARIMA model parameters for the specified stock symbol.
    """
    if input_dir is None:
        input_dir = Path(cf.MODEL_PARAMATERS)
    else:
        input_dir = Path(input_dir)
    
    input_file = input_dir / f'{stock_symbol}_arima_parameters.pkl'
    
    with open(input_file, 'rb') as f:
        parameters = pickle.load(f)
    
    return parameters

#pipeline:
def load_garch_parameters(stock_symbol, input_dir=None):
    """
    Loads GARCH model parameters for specified stock symbol from pickle file in input directory.
    Parameters:
        stock_symbol (str): Symbol of the stock for which to load GARCH parameters.
        input_dir (str or Path, optional): Directory where GARCH parameter pickle files are
                                           stored. If None, defaults to config.MODEL_PARAMATERS.
    Returns:
        parameters (dict): Dictionary containing GARCH model parameters for the specified stock symbol.
    """
    if input_dir is None:
        input_dir = Path(cf.MODEL_PARAMATERS)
    else:
        input_dir = Path(input_dir)
    
    input_file = input_dir / f'{stock_symbol}_garch_parameters.pkl'
    
    with open(input_file, 'rb') as f:
        parameters = pickle.load(f)
    
    return parameters

def model_pipeline(all_results=None):
    """
    Executes the full modeling pipeline for each stock symbol, including fitting ARIMA and GARCH models,
    generating forecasts, evaluating performance, and saving results.
    Parameters:
        all_results (dict, optional): If provided, should be a dictionary to store evaluation metrics and plot
                                        filenames for each stock symbol.
    Returns:
        Either:
        - all_results (dict): Dictionary containing evaluation metrics and plot filenames for each stock symbol.
        - None: If the function is modified to save results to files instead of returning them.
    """

    cf.ensure_directories()
    returns_data = load_data('log_returns_data.csv')
    train_data, test_data = train_test_split(returns_data)
    
    all_results = {}
    
    for stock_symbol in cf.SYMBOLS:
        print(f"{stock_symbol}:")
        
        arima_model = fit_arima(train_data, stock_symbol)
        forecast_ARIMA = forecast_arima(arima_model)
        arima_metrics, arima_params = evaluate_arima(arima_model, forecast_ARIMA, stock_symbol, test_data)
        print(f"ARIMA Metrics: {arima_metrics}")
        
        arima_forecast_df = convert_forecast_to_df(forecast_ARIMA, returns_data, stock_symbol)
        arima_fig = plot_arima_forecast(returns_data, arima_forecast_df, stock_symbol)
        arima_plot_name = f'{stock_symbol}_arima_forecast.png'
        save_plots(arima_fig, directory=cf.MODELS_DIR, filename=arima_plot_name, showfile=False)
        plt.close(arima_fig)
        
        garch_results, _ = fit_garch(train_data, test_data, stock_symbol)
        garch_params_dict = extract_garch_params(garch_results)
        garch_metrics, garch_params = evaluate_garch(garch_params_dict, test_data, stock_symbol)
        print(f"GARCH Metrics: {garch_metrics}")
        
        save_parameters_pickle(arima_params, stock_symbol, garch_params, output_dir=cf.MODEL_PARAMATERS)
        
        garch_forecast_fig = plot_garch_forecast_volatility(train_data, test_data, garch_params_dict, stock_symbol)
        garch_forecast_plot_name = f'{stock_symbol}_{cf.ARCH_TYPE}_forecast_volatility.png'
        save_plots(garch_forecast_fig, directory=cf.MODELS_DIR, filename=garch_forecast_plot_name, showfile=False)
        plt.close(garch_forecast_fig)
        
        all_results[stock_symbol] = {
            'arima_metrics': arima_metrics,
            'garch_metrics': garch_metrics,
            'plots': [arima_plot_name, garch_forecast_plot_name]
        }
    
    return all_results if all_results is not None else None

def main():
    model_pipeline()
    
if __name__ == "__main__":
    main()