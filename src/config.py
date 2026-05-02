from datetime import datetime, timedelta
from pathlib import Path


# Path configuration:
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
PREPROCESSING_DIR = FIGURES_DIR / "preprocessing"
TIME_SERIES_DIR = FIGURES_DIR / "time_series"
MODELS_DIR = FIGURES_DIR / "models"
MONTE_CARLO_DIR = FIGURES_DIR / "monte_carlo"
COMPARISON_DIR = FIGURES_DIR / "comparison"


def ensure_directories():
   """
   Ensure that all necessary directories exist. If not, create them.
   """
   for directory in [RAW_DATA, PROCESSED_DATA, FIGURES_DIR, PREPROCESSING_DIR, TIME_SERIES_DIR, MODELS_DIR, MONTE_CARLO_DIR, COMPARISON_DIR]:
       directory.mkdir(parents=True, exist_ok=True)


# Stock configuration: Balanced portfolio of 5 stocks across tech, consumer, energy and finance
SYMBOLS = ['AMD', 'GOOGL', 'JPM', 'XOM', 'KO']
STOCK_EXCHANGE_TIMEZONE = 'NYSE' # change to whatever is relevant for stocks chosen (NASDAQ and NYSE use same).
START_DATE = datetime(2020, 1, 1) # 5+ years of data for MC simulations
END_DATE = datetime.now()
INTERVAL = '1d' # Daily data
NUM_STOCKS = len(SYMBOLS)

# Time series parameters:

ROLLING_WINDOW = 21 # 21 trading days in a month, I think it will be a good window.
MIN_WINDOW = 10 # Minimum window size for rolling calcs.
ANOMALY_THETA_THRESHOLD = 3 #threshold for close-open anomaly detection in terms of standard deviations.
KURTOSIS_THRESHOLD = 3 # threshold for excess kurtosis to flag potential outliers in log returns.
ANNUALISED_WINDOW = 252 # Num of trading days in a year (for annualising volatility)



# Monte Carlo Simulation parameters (TBD for now, not until data pipeline is done):




#Model parameters arima, garch etc(TBD for now, not until data pipeline is done):
