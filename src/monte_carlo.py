import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import config as cf
from src.data_pipeline import load_data, save_plots
from src.time_series import correlation_matrix, covariance_matrix
from src.models import load_garch_parameters


def gbm_simulation(last_price, mu, sigma, forecast_steps, num_simulations, rng=None):
    """
    Simulates stock price paths using Geometric Brownian Motion models alongside Black-Scholes assumptions.
    Parameters:
        last_price (float): The last observed stock price to start simulations from.
        mu (float): The expected return (drift) of the stock.
        sigma (float): The volatility of the stock.
        forecast_steps (int): The number of time steps to simulate into the future.
        num_simulations (int): The number of simulated paths to generate.
    returns:
        simulated_paths (np.ndarray): An array of shape (num_simulations, forecast_steps) containing the simulated price paths.
    """
    if rng is None:
        rng = np.random.default_rng(cf.RANDOM_SEED) # create a random number generator instance with the specified seed for reproducibility
    dt = 1 # daily time steps
    simulated_paths = np.zeros((num_simulations, forecast_steps))
    simulated_paths[:, 0] = last_price

    for t in range(1, forecast_steps):
        # GBM: S(t+ 1) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z))
        z = rng.normal(size=num_simulations) # standard normal random variables
        simulated_paths[:, t] = simulated_paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    return simulated_paths
    
def simulate_price_paths(model_parameters, correlation_matrix, rng, forecast_steps, dist=cf.GARCH_DIST):
    """
    Simulates future price paths based on the specified model and its parameters.
    This assumes a constant correlation between assets for simplicity.
    Parameters:
        model_parameters (dict): The fitted model parameters to use for simulation.
        correlation_matrix (np.ndarray): A matrix containing the correlations between assets.
        forecast_steps (int): The number of time steps to simulate into the future.
        rng (np.random.default_rng): Random number generator instance for reproducibility.
        dist (str): The distribution to use for simulating returns. Options are 'Normal' or 'StudentsT'.
    Returns:
        simulated_paths (np.ndarray): An array of shape (num_simulations, forecast_steps) containing the simulated price paths.
    """
    if dist not in ['Normal', 'StudentsT']:
        raise ValueError("Invalid distribution specified. Use 'Normal' or 'StudentsT'.")

    n_stocks = cf.NUM_STOCKS
    L = np.linalg.cholesky(correlation_matrix) # Cholesky decomposition to get lower triangular matrix for correlation
    path = np.zeros((forecast_steps, n_stocks))

    # Extract parameters - now model_parameters is a dict for ONE stock per simulation iteration
    sigma = model_parameters['last_sigma']
    epsilon = model_parameters['last_epsilon']
    omega = model_parameters['omega']
    alpha = model_parameters['alpha']
    beta = model_parameters['beta']
    mu = model_parameters['mu']
    dof = model_parameters.get('dof', 5)  # default dof if not specified

    for t in range(forecast_steps):
        if dist == 'Normal':
            distribution = rng.normal(size=n_stocks) # independent standard normal random variables)
        elif dist == 'StudentsT':
            distribution = rng.standard_t(dof, size=n_stocks)
        
        distribution = np.clip(distribution, -3, 3) # clip extreme values to prevent tail dominance in simulations
        random_shocks = L @ distribution # introduce correlation
        
        sigma2 = omega + (alpha * epsilon**2) + (beta * sigma**2)
        sigma = np.sqrt(sigma2)
        epsilon = random_shocks[0] * sigma  # or handle multiple stocks if needed
        path[t, 0] = mu + epsilon # simulate return for this time step
    
    return path
    
def monte_carlo_simulation(model_parameters, correlation_matrix, forecast_steps, rng, dist=cf.GARCH_DIST, num_simulations=cf.NUM_SIMULATIONS):
    """
    Main function to run Monte Carlo simulations based on the specified model parameters.
    
    Parameters:
        model_params (dict): A dictionary containing the fitted model parameters.
        last_price (float): The last observed stock price to start simulations from.
        forecast_steps (int): The number of time steps to simulate into the future.
        num_simulations (int): The number of simulated paths to generate.
        
    Returns:
        simulated_paths (np.ndarray): An array of shape (num_simulations, forecast_steps) containing the simulated price paths.
    """
    n_stocks = cf.NUM_STOCKS
    simulated_paths = np.zeros((num_simulations, forecast_steps, n_stocks))

    for sim in range(num_simulations):
        simulated_paths[sim] = simulate_price_paths(
                                                    model_parameters=model_parameters,
                                                    correlation_matrix=correlation_matrix,
                                                    forecast_steps=forecast_steps,
                                                    rng=rng,
                                                    dist=dist
        )

    return simulated_paths

def convert_returns_to_prices(simulated_returns, last_price):
    """
    Converts simulated log returns to price paths starting from the last observed price.
    
    Parameters:
        simulated_returns (np.ndarray): An array of shape (num_simulations, forecast_steps, n_stocks) containing the simulated log returns.
        last_price (float): The last observed stock price to start simulations from.
        
    Returns:
        simulated_prices (np.ndarray): An array of shape (num_simulations, forecast_steps, n_stocks) containing the simulated price paths.
    """
    return last_price * np.exp(np.cumsum(simulated_returns, axis=1))

def plot_simulated_prices(simulated_prices, stock_symbol, num_paths=20, ax=None):
    """
    Plots the simulated price paths.
    
    Parameters:
        simulated_prices (np.ndarray): An array of shape (num_simulations, forecast_steps) containing the simulated price paths.
        stock_symbol (str): The stock symbol for labeling the plot.
        num_paths (int): The number of simulated paths to plot for clarity.
        ax (matplotlib.axes.Axes, optional): An optional Axes object to plot on. If None, a new figure
                                             and axes will be created.
    returns:
        fig (matplotlib.figure.Figure): The figure object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()
    
    for i in range(min(num_paths, simulated_prices.shape[0])):
        ax.plot(simulated_prices[i, :])
    ax.set_title(f'Simulated Price Paths for {stock_symbol}')
    ax.set_xlabel('Forecast Time Steps (Days)')
    ax.set_ylabel('Price')
    return fig

def var_es_calculations(simulated_prices, initial_price, confidence_level=0.95):
    """
    Calculates Value at Risk (VaR) and Expected Shortfall (ES) from the simulated price paths.
    
    Parameters:
        simulated_prices (np.ndarray): An array of shape (num_simulations, forecast_steps) containing the simulated price paths.
        initial_price (float): The initial stock price to calculate returns from.
        confidence_level (float): The confidence level for VaR calculation (e.g., 0.95 for 95% confidence).
        
    Returns:
        var (float): The Value at Risk at the specified confidence level.
        es (float): The Expected Shortfall at the specified confidence level.
    """
    # Calculate returns from simulated prices
    simulated_returns = (simulated_prices[:, -1] - initial_price) / initial_price
    losses = -simulated_returns  # Losses are negative returns
    var = np.percentile(losses, (confidence_level * 100))
    es = losses[losses >= var].mean()  # Average loss beyond the VaR threshold
    return {
        'VaR': var,
        'ES': es,
        'portfolio_returns': simulated_returns,
        'losses': losses
    }

def plot_simulated_distribution(simulated_prices, initial_price, stock_symbol, bins=50, ax=None):
    """
    Plots the distribution of simulated returns at the final time step.
    
    Parameters:
        simulated_prices (np.ndarray): An array of shape (num_simulations, forecast_steps) containing the simulated price paths.
        initial_price (float): The initial stock price to calculate returns from.
        stock_symbol (str): The stock symbol for labeling the plot.
        bins (int): The number of bins to use in the histogram.
        ax (matplotlib.axes.Axes, optional): An optional Axes object to plot on. If None, a new figure
                                             and axes will be created.
    returns:
        fig (matplotlib.figure.Figure): The figure object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    # Calculate returns from simulated prices
    sns.histplot(simulated_prices[:, -1], bins=bins, kde=True, ax=ax)
    ax.axvline(initial_price, color='red', linestyle='--')  # Add a vertical line at initial price for reference
    ax.set_title(f'Distribution of Simulated Prices for {stock_symbol}')
    ax.set_xlabel('Simulated Final Prices')
    ax.set_ylabel('Frequency')
    return fig

def calculate_confidence_intervals(simulated_prices, lower_percentile=cf.LOWER_PERCENTILE, upper_percentile=cf.UPPER_PERCENTILE):
    """
    Calculates confidence intervals for the simulated price paths at each time step.
    
    Parameters:
        simulated_prices (np.ndarray): An array of shape (num_simulations, forecast_steps) containing the simulated price paths.
        lower_percentile (float): The lower percentile for the confidence interval
        upper_percentile (float): The upper percentile for the confidence interval 
        
    Returns:
        ci_lower (np.ndarray): An array of shape (forecast_steps,) containing the lower bound of the confidence interval at each time step.
        ci_upper (np.ndarray): An array of shape (forecast_steps,) containing the upper bound of the confidence interval at each time step.
    """
    median_path = np.median(simulated_prices, axis=0)
    mean_path = np.mean(simulated_prices, axis=0)
    ci_lower = np.percentile(simulated_prices, lower_percentile, axis=0)
    ci_upper = np.percentile(simulated_prices, upper_percentile, axis=0)
    return {
        'median': median_path,
        'mean': mean_path,
        'lower_band': ci_lower,
        'upper_band': ci_upper
    }

def plot_confidence_intervals(simulated_prices, confidence_dictionary, stock_symbol, ax=None):
    """
    Plots the simulated price paths along with confidence intervals.
    
    Parameters:
        simulated_prices (np.ndarray): An array of shape (num_simulations, forecast_steps) containing the simulated price paths.
        confidence_dictionary (dict): A dictionary containing the confidence interval bounds.
        stock_symbol (str): The stock symbol for labeling the plot.
        ax (matplotlib.axes.Axes, optional): An optional Axes object to plot on. If None, a new figure
                                             and axes will be created.
    returns:
        fig (matplotlib.figure.Figure): The figure object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()
    
    steps = np.arange(len(confidence_dictionary['median']))
    ax.plot(steps, confidence_dictionary['median'], label='Median Path', color='blue')
    ax.plot(steps, confidence_dictionary['mean'], label='Mean Path', color='orange')
    ax.fill_between(steps, confidence_dictionary['lower_band'], confidence_dictionary['upper_band'], color='green', alpha=0.3,
                    label=f'{cf.LOWER_PERCENTILE} - {cf.UPPER_PERCENTILE}% Confidence Interval')
    ax.set_title(f'Confidence Intervals for {stock_symbol}')
    ax.set_xlabel('Forecast Time Steps (Days)')
    ax.set_ylabel('Price')
    return fig

def generate_random_weights(num_assets, rng):
    """
    Generates a random set of portfolio weights that sum to 1.
    
    Parameters:
        num_assets (int): The number of assets in the portfolio.
        rng (np.random.default_rng): Random number generator instance for reproducibility.
        
    Returns:
        weights (np.ndarray): An array of shape (num_assets,) containing the random weights for each asset.
    """
    weights = rng.random(num_assets)
    weights /= np.sum(weights)  # Normalize to sum to 1
    return weights
    
def calculate_portfolio_returns(simulated_prices, weights):
    """
    Calculates the portfolio returns from the simulated price paths and given weights.
    
    Parameters:
        simulated_prices (np.ndarray): An array of shape (num_simulations, forecast_steps, n_stocks) containing the simulated price paths.
        weights (np.ndarray): An array of shape (n_stocks,) containing the weights for each stock in the portfolio.
        
    Returns:
        portfolio_returns (np.ndarray): An array of shape (num_simulations, forecast_steps) containing the simulated portfolio returns (simple returns).
    """
    # Calculate returns for each stock
    final_prices = simulated_prices[:, -1, :]  # Get the final prices for each simulation and stock
    initial_prices = simulated_prices[:, 0, :]  # Get the initial prices for each simulation and stock
    returns = (final_prices - initial_prices) / initial_prices  # Calculate returns for each stock
    portfolio_returns = returns @ weights  # Calculate portfolio returns as weighted sum
    return portfolio_returns

def portfolio_statistics(weights, mean_returns, covariance_matrix, risk_free_rate=cf.RISK_FREE_RATE):
    """
    Calculates portfolio return, risk and Sharpe ratio for given weights, mean returns and covariance matrix.
    Parameters:
        weights (np.ndarray): An array of shape (n_stocks,) containing the weights for each stock in the portfolio.
        mean_returns (np.ndarray): An array of shape (n_stocks,) containing the mean returns for each stock.
        covariance_matrix (np.ndarray): An array of shape (n_stocks, n_stocks) containing the covariance matrix of returns.
        risk_free_rate (float): The risk-free rate to use for Sharpe ratio calculation.
    Returns:
        portfolio_stats (dict): A dictionary containing the portfolio return, risk and Sharpe ratio.
    """
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_risk = np.sqrt(weights.T @ covariance_matrix @ weights)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk != 0 else 0

    return {
        "return": portfolio_return,
        "risk": portfolio_risk,
        "sharpe_ratio": sharpe_ratio
    }

def efficient_frontier(mean_returns, covariance_matrix, rng, num_portfolios, risk_free_rate=cf.RISK_FREE_RATE):
    """
    Generates the efficient frontier for a given set of assets.
    Parameters:
        mean_returns (np.ndarray): An array of shape (n_stocks,) containing the mean returns for each stock.
        covariance_matrix (np.ndarray): An array of shape (n_stocks, n_stocks) containing the covariance matrix of returns.
        rng (np.random.default_rng): Random number generator instance for reproducibility.
        num_portfolios (int): The number of portfolios to generate along the efficient frontier.
        risk_free_rate (float): The risk-free rate to use for Sharpe ratio calculation.
    Returns:
        efficient_portfolios (list): A list of dictionaries containing the weights, return, risk and Sharpe ratio for each portfolio on the efficient frontier.
    """
    n_stocks = len(mean_returns)
    portfolio_returns = []
    portfolio_risks = []
    portfolio_sharpe_ratios = []
    portfolio_weights = []

    for _ in range(num_portfolios):
        weights = generate_random_weights(n_stocks, rng)
        portfolio_stats = portfolio_statistics(weights, mean_returns, covariance_matrix, risk_free_rate)
        portfolio_returns.append(portfolio_stats["return"])
        portfolio_risks.append(portfolio_stats["risk"])
        portfolio_sharpe_ratios.append(portfolio_stats["sharpe_ratio"])
        portfolio_weights.append(weights)

    return {
        "returns": np.array(portfolio_returns),
        "risks": np.array(portfolio_risks),
        "sharpe_ratios": np.array(portfolio_sharpe_ratios),
        "weights": np.array(portfolio_weights)
    }

def plot_efficient_frontier(efficient_frontier_data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()
    scatter = ax.scatter(efficient_frontier_data['risks'], efficient_frontier_data['returns'],
                         c=efficient_frontier_data['sharpe_ratios'], alpha=0.5)
    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Portfolio Risk')
    ax.set_ylabel('Portfolio Return')
    fig.colorbar(scatter, label='Sharpe Ratio')
    return fig

def optimise_portfolio(mean_returns, covariance_matrix, risk_free_rate=cf.RISK_FREE_RATE):
    """
    Optimises the portfolio to find the weights that maximise the Sharpe ratio.
    Parameters:
        mean_returns (np.ndarray): An array of shape (n_stocks,) containing the mean returns for each stock.
        covariance_matrix (np.ndarray): An array of shape (n_stocks, n_stocks) containing the covariance matrix of returns.
        risk_free_rate (float): The risk-free rate to use for Sharpe ratio calculation.
    Returns:
        optimal_portfolio (dict): A dictionary containing the optimal weights, return, risk and Sharpe ratio for the optimal portfolio.
    """
    n_stocks = len(mean_returns)
    initial_weights = np.ones(n_stocks) / n_stocks  # Start with equal weights

    def negative_sharpe_ratio(weights):
        stats = portfolio_statistics(weights, mean_returns, covariance_matrix, risk_free_rate)
        return -stats["sharpe_ratio"]  # maximise sharpe ratio therefore minimise negative sharpe

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # sum of weights = 1 constraint
    bounds = tuple((0, 1) for _ in range(n_stocks))  # weights cannot be in negative or greater than 1

    result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = result.x
    optimal_stats = portfolio_statistics(optimal_weights, mean_returns, covariance_matrix, risk_free_rate)

    return {
        "optimal weights": optimal_weights,
        "optimal return": optimal_stats["return"],
        "optimal risk": optimal_stats["risk"],
        "optimal sharpe ratio": optimal_stats["sharpe_ratio"]
    }

def monte_carlo_pipeline():
    """
    Main pipeline for Monte Carlo simulation of stock price paths based on fitted models.
    """
    cf.ensure_directories()
    returns_data = load_data("log_returns_data.csv")
    price_data = load_data("cleaned_portfolio_data.csv")
    cov_matrix = covariance_matrix(returns_data, return_matrix=True) # covariance matrix of log returns
    rng = np.random.default_rng(cf.RANDOM_SEED) # create a random number generator instance with the specified seed for reproducibility
    all_simulated_prices = {}
    mean_returns_list = []
    
    for stock_symbol in cf.SYMBOLS:
        plot_directory = cf.MONTE_CARLO_DIR / stock_symbol
        column_name = f'{stock_symbol}_Log_Returns'
        stock_idx = cf.SYMBOLS.index(stock_symbol)
        gbm_volatility = np.sqrt(cov_matrix[stock_idx, stock_idx])
        print(f"Simulating price paths for {stock_symbol}")
        last_return = returns_data[column_name].iloc[-1] # get last observed log return for the stock
        last_price = price_data[stock_symbol].iloc[-1] # get last observed price for the stock
        model_params = load_garch_parameters(stock_symbol)
        mean_returns = returns_data[column_name].mean()
        mean_returns_list.append(mean_returns)

        # GBM based Monte Carlo simulation: assumes constant volatility and drift
        gbm_simulated_prices = gbm_simulation(
            last_price=last_price,
            mu=mean_returns,
            sigma=gbm_volatility,
            forecast_steps=cf.FORECAST_STEPS,
            num_simulations=cf.NUM_SIMULATIONS,
            rng=rng
        )

        # Plot GBM simulated price paths for this stock:
        fig = plot_simulated_prices(gbm_simulated_prices, stock_symbol, num_paths=50) # plot only 5 paths for clarity
        save_plots(fig, directory=plot_directory, filename=f'{stock_symbol}_GBM_monte_carlo_simulation.png', showfile=False)
        plt.close(fig)

        #GBM var and es calculations:
        var_es_results = var_es_calculations(gbm_simulated_prices, last_price, confidence_level=cf.CONFIDENCE_LEVEL)
        print(f"GBM VaR at {cf.CONFIDENCE_LEVEL*100}% confidence for {stock_symbol}: {var_es_results['VaR']:.4f}")
        print(f"GBM Expected Shortfall at {cf.CONFIDENCE_LEVEL*100}% confidence for {stock_symbol}: {var_es_results['ES']:.4f}")

        # Garch based Monte Carlo simulation: assumes const mean and time varying volatility
        simulated_returns = monte_carlo_simulation(
            model_parameters=model_params,
            correlation_matrix=correlation_matrix(returns_data, return_matrix=True), # correlation matrix of log returns
            forecast_steps=cf.FORECAST_STEPS,
            rng=rng,
            dist=cf.GARCH_DIST,
            num_simulations=cf.NUM_SIMULATIONS,
        )
        simulated_prices = convert_returns_to_prices(simulated_returns, last_price)
        simulated_prices = simulated_prices[:, :, 0] # extract the simulated price paths for this stock from the 3D array
        all_simulated_prices[stock_symbol] = simulated_prices

        #Plot price paths for this stock:
        fig = plot_simulated_prices(simulated_prices, stock_symbol, num_paths=50) # plot only 5 paths for clarity
        save_plots(fig, directory=plot_directory, filename=f'{stock_symbol}_{cf.GARCH_DIST}_monte_carlo_simulation.png', showfile=False)
        plt.close(fig)

        #Value at Risk and Expected Shortfall calculations:
        var_es_results = var_es_calculations(simulated_prices, last_price, confidence_level=cf.CONFIDENCE_LEVEL)
        print(f"VaR at {cf.CONFIDENCE_LEVEL*100}% confidence for {stock_symbol}: {var_es_results['VaR']:.4f}")
        print(f"Expected Shortfall at {cf.CONFIDENCE_LEVEL*100}% confidence for {stock_symbol}: {var_es_results['ES']:.4f}")
        
        #Simulated price distribution plot
        fig = plot_simulated_distribution(simulated_prices, last_price, stock_symbol)
        save_plots(fig, directory=plot_directory, filename=f'{stock_symbol}_{cf.GARCH_DIST}_simulated_price_distribution.png', showfile=False)
        plt.close(fig)

        #plot confidence intervals for this stock:
        confidence_dict = calculate_confidence_intervals(simulated_prices)
        fig = plot_confidence_intervals(simulated_prices, confidence_dict, stock_symbol)
        save_plots(fig, directory=plot_directory, filename=f'{stock_symbol}_{cf.GARCH_DIST}_confidence_intervals.png', showfile=False)
        plt.close(fig)

    #Portfolio optimisation:
    mean_returns = np.array(mean_returns_list)

    # Generate efficient frontier
    efficient_frontier_data = efficient_frontier(mean_returns, cov_matrix, rng, num_portfolios=cf.FRONTIER_PORTFOLIOS)
    fig = plot_efficient_frontier(efficient_frontier_data)
    save_plots(fig, directory=cf.COMPARISON_DIR, filename=f'efficient_frontier.png', showfile=False)
    plt.close(fig)

    # Optimise portfolio
    optimal_portfolio = optimise_portfolio(mean_returns, cov_matrix)
    for i, symbol in enumerate(cf.SYMBOLS):
        print(f"Optimal weight for {symbol}: {optimal_portfolio['optimal weights'][i]:.4f}")

    print(f"Optimal portfolio return: {optimal_portfolio['optimal return']:.4f}")
    print(f"Optimal portfolio risk: {optimal_portfolio['optimal risk']:.4f}")
    print(f"Optimal portfolio Daily Sharpe ratio: {optimal_portfolio['optimal sharpe ratio']:.4f}")  
    print(f"Optimal portfolio Annualised Sharpe ratio: {(optimal_portfolio['optimal sharpe ratio'] * np.sqrt(cf.ANNUALISED_WINDOW)):.4f}") 

def main():
    monte_carlo_pipeline()

if __name__ == "__main__":
    main()