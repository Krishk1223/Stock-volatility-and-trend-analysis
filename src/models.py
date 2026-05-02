import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import config as cf
from src.data_pipeline import load_data
from pmdarima import auto_arima
from arch import 
