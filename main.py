import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from statsmodels.tsa.stattools import coint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
from datetime import datetime, timedelta
import warnings

# Visual setup
plt.style.use('ggplot')
warnings.filterwarnings('ignore')

# -----------------------------
# CONFIGURATION
# -----------------------------
API_KEY = os.getenv("MASSIVE_API_KEY")  # Make sure to set this in your .env or GitHub Secrets
BASE_URL = "https://files.massive.com"  # Massive.com endpoint for stock CSVs

# Folder structure
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/data", exist_ok=True)

# Pairs for analysis
PAIRS = {"Energy": ["XOM", "CVX"], "Tech": ["NVDA", "AMD"]}

# Date range for last 2 years
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=730)

# -----------------------------
# FUNCTION TO PULL DATA
# -----------------------------
def fetch_massive_data(symbol: str) -> pd.DataFrame:
    """
    Fetches historical daily stock data from massive.com for the given symbol.
    """
    headers = {"x-api-key": API_KEY}
    # Assuming CSV files are available like: https://files.massive.com/stocks/XOM.csv
    url = f"{BASE_URL}/stocks/{symbol}.csv"

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching {symbol} data: {response.status_code}")
    
    # Read CSV into DataFrame
    df = pd.read_csv(pd.compat.StringIO(response.text), parse_dates=['Date'])
    
    # Filter last 2 years
    df = df[(df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)]
    df = df.sort_values('Date').set_index('Date')
    
    return df[['Close']]  # Keep only the Close price

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def run():
    print("🚀 Running Pairs Trading Analysis...")
    
    for name, syms in PAIRS.items():
        try:
            # 1. Fetch Real Data
            df_a, df_b = fetch_massive_data(syms[0]), fetch_massive_data(syms[1])
            df = pd.DataFrame({'A': df_a['Close'], 'B': df_b['Close']}).dropna()

            # 2. MATH IA: Cointegration
            _, p_val, _ = coint(df['A'], df['B'])

            # 3. Spread Calculation
            ratio = df['A'].mean() / df['B'].mean()
            df['Spread'] = df['A'] - (ratio * df['B'])
            df['Z_Score'] = (df['Spread'] - df['Spread'].mean()) / df['Spread'].std()

            # 4. CS EE: Machine Learning
            df['Target'] = np.where(df['Z_Score'].shift(-1) < df['Z_Score'], 1, 0)
            df = df.dropna()
            X_train, X_test, y_train, y_test = train_test_split(df[['Z_Score']], df['Target'], test_size=0.2, shuffle=False)

            model = xgb.XGBClassifier(eval_metric='logloss').fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))

            print(f"Pair: {name} | Coint p-value: {p_val:.4f} | ML Accuracy: {acc:.2%}")

            # 5. Save Graphics
            plt.figure(figsize=(10, 4))
            plt.plot(df['Z_Score'], label='Z-Score', color='blue')
            plt.axhline(2, color='red', ls='--')
            plt.axhline(-2, color='green', ls='--')
            plt.title(f"Mean Reversion Signal: {name}")
            plt.savefig(f"results/plots/{name}_zscore.png")
            plt.close()

            df.to_csv(f"results/data/{name}_results.csv")
        
        except Exception as e:
            print(f"Error processing pair {name}: {e}")

if __name__ == "__main__":
    run()
