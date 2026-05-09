from src import config as cf
from src.data_pipeline import data_preprocessing_pipeline
from src.time_series import time_series_pipeline
from src.models import model_pipeline
from src.monte_carlo import monte_carlo_pipeline


def main():
    cf.ensure_directories() # Ensure all necessary directories exist before starting the pipeline.
    print("RUNNING STOCK ANALYSIS PIPELINE")

    # Step 1: Data Collection and Preprocessing
    print("\nStarting data preprocessing pipeline:")
    data_preprocessing_pipeline()
    print("Data preprocessing completed successfully.")

    # Step 2: Time Series Analysis
    print("\nStarting time series analysis pipeline:")
    time_series_pipeline()
    print("Time series analysis completed successfully.")

    # Step 3: Modeling and Forecasting
    print("\nStarting modeling and forecasting pipeline:") 
    model_pipeline()
    print("Modeling and forecasting completed successfully.")

    # Step 4: Monte Carlo Simulation and Portfolio Optimisation
    print("\nStarting Monte Carlo simulation and portfolio optimisation pipeline:")
    monte_carlo_pipeline()
    print("\nMonte Carlo simulation and portfolio optimisation completed successfully.")

if __name__ == "__main__":
    main()
