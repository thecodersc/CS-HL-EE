import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from statsmodels.tsa.stattools import coint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yfinance as yf
from datetime import datetime, timedelta
import warnings

# Visual setup
plt.style.use('ggplot')
warnings.filterwarnings('ignore')

# -----------------------------
# CONFIGURATION
# -----------------------------
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
def fetch_stock_data(symbol: str) -> pd.DataFrame:
    """
    Fetches historical daily stock data from Yahoo Finance for the given symbol.
    """
    print(f"  Downloading {symbol} data from Yahoo Finance...")
    
    try:
        # Download data using yfinance
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=START_DATE, end=END_DATE)
        
        if df.empty:
            raise Exception(f"No data returned for {symbol}")
        
        # Keep only the Close price
        df = df[['Close']].copy()
        df.index.name = 'Date'
        
        print(f"  ✓ Downloaded {len(df)} trading days")
        return df
        
    except Exception as e:
        raise Exception(f"Error fetching {symbol}: {str(e)}")

# -----------------------------
# BASELINE: STATISTICAL-ONLY STRATEGY
# -----------------------------
def statistical_baseline_strategy(df):
    """
    Traditional statistical pairs trading using Z-score thresholds only.
    This is your BASELINE to compare against XGBoost.
    """
    # Generate signals: Buy when Z < -2, Sell when Z > 2
    df['Stat_Signal'] = 0
    df.loc[df['Z_Score'] < -2, 'Stat_Signal'] = 1  # Buy signal
    df.loc[df['Z_Score'] > 2, 'Stat_Signal'] = -1  # Sell signal
    
    # Create target: did the spread mean-revert in next period?
    df['Stat_Target'] = np.where(df['Z_Score'].shift(-1) < df['Z_Score'], 1, 0)
    
    return df

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def run():
    print("=" * 60)
    print("PAIRS TRADING ANALYSIS: Statistical vs ML-Enhanced")
    print("Research Question: Does XGBoost enhance cointegration trading?")
    print("=" * 60)
    
    all_results = []
    
    for name, syms in PAIRS.items():
        try:
            print(f"\n{'='*60}")
            print(f"Processing {name} pair: {syms[0]} vs {syms[1]}")
            print('='*60)
            
            # 1. Fetch Real Data
            df_a = fetch_stock_data(syms[0])
            df_b = fetch_stock_data(syms[1])
            
            df = pd.DataFrame({'A': df_a['Close'], 'B': df_b['Close']}).dropna()
            print(f"\nCombined dataset: {len(df)} trading days")
            
            # 2. MATH IA: Cointegration Test
            score, p_val, crit_values = coint(df['A'], df['B'])
            is_cointegrated = p_val < 0.05
            
            print(f"\n--- COINTEGRATION ANALYSIS ---")
            print(f"Test Statistic: {score:.4f}")
            print(f"P-value: {p_val:.4f}")
            print(f"Critical values (1%, 5%, 10%): {crit_values}")
            print(f"Cointegrated at 5% level: {'YES' if is_cointegrated else 'NO'}")
            
            # 3. Spread Calculation
            ratio = df['A'].mean() / df['B'].mean()
            df['Spread'] = df['A'] - (ratio * df['B'])
            df['Z_Score'] = (df['Spread'] - df['Spread'].mean()) / df['Spread'].std()
            
            print(f"\nSpread Statistics:")
            print(f"  Mean: {df['Spread'].mean():.4f}")
            print(f"  Std Dev: {df['Spread'].std():.4f}")
            print(f"  Min Z-Score: {df['Z_Score'].min():.2f}")
            print(f"  Max Z-Score: {df['Z_Score'].max():.2f}")
            
            # 4. BASELINE: Statistical Strategy
            df = statistical_baseline_strategy(df)
            
            # 5. CS EE: XGBoost Enhanced Strategy
            df['ML_Target'] = np.where(df['Z_Score'].shift(-1) < df['Z_Score'], 1, 0)
            df_clean = df.dropna()
            
            # Split data: 80% train, 20% test
            split_idx = int(len(df_clean) * 0.8)
            train_data = df_clean.iloc[:split_idx]
            test_data = df_clean.iloc[split_idx:]
            
            print(f"\nDataset Split:")
            print(f"  Training: {len(train_data)} days")
            print(f"  Testing: {len(test_data)} days")
            
            # Train XGBoost
            X_train = train_data[['Z_Score']]
            y_train = train_data['ML_Target']
            X_test = test_data[['Z_Score']]
            y_test = test_data['ML_Target']
            
            model = xgb.XGBClassifier(
                eval_metric='logloss',
                max_depth=3,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_ml = model.predict(X_test)
            
            # Statistical baseline predictions on test set
            y_pred_stat = test_data['Stat_Signal'].apply(lambda x: 1 if x != 0 else 0)
            
            # 6. COMPARISON METRICS
            ml_accuracy = accuracy_score(y_test, y_pred_ml)
            ml_precision = precision_score(y_test, y_pred_ml, zero_division=0)
            ml_recall = recall_score(y_test, y_pred_ml, zero_division=0)
            ml_f1 = f1_score(y_test, y_pred_ml, zero_division=0)
            
            stat_accuracy = accuracy_score(y_test, y_pred_stat)
            stat_precision = precision_score(y_test, y_pred_stat, zero_division=0)
            stat_recall = recall_score(y_test, y_pred_stat, zero_division=0)
            stat_f1 = f1_score(y_test, y_pred_stat, zero_division=0)
            
            improvement = ((ml_accuracy - stat_accuracy) / stat_accuracy * 100) if stat_accuracy > 0 else 0
            
            print(f"\n--- RESULTS COMPARISON ---")
            print(f"\nSTATISTICAL BASELINE (Z-Score Thresholds):")
            print(f"  Accuracy:  {stat_accuracy:.2%}")
            print(f"  Precision: {stat_precision:.2%}")
            print(f"  Recall:    {stat_recall:.2%}")
            print(f"  F1-Score:  {stat_f1:.2%}")
            
            print(f"\nXGBOOST ENHANCED MODEL:")
            print(f"  Accuracy:  {ml_accuracy:.2%}")
            print(f"  Precision: {ml_precision:.2%}")
            print(f"  Recall:    {ml_recall:.2%}")
            print(f"  F1-Score:  {ml_f1:.2%}")
            
            print(f"\n{'='*60}")
            print(f"IMPROVEMENT: {improvement:+.2f}%")
            print(f"CONCLUSION: XGBoost {'ENHANCES' if improvement > 0 else 'DOES NOT ENHANCE'} the strategy")
            print(f"{'='*60}")
            
            # Save detailed results
            results = {
                'Pair': name,
                'Stock_A': syms[0],
                'Stock_B': syms[1],
                'Data_Points': len(df_clean),
                'Test_Period_Days': len(test_data),
                'Cointegration_PValue': p_val,
                'Is_Cointegrated': is_cointegrated,
                'Stat_Accuracy': stat_accuracy,
                'Stat_Precision': stat_precision,
                'Stat_F1': stat_f1,
                'ML_Accuracy': ml_accuracy,
                'ML_Precision': ml_precision,
                'ML_F1': ml_f1,
                'Improvement_Percent': improvement
            }
            all_results.append(results)
            
            # 7. Save summary to file
            with open(f"results/data/{name}_summary.txt", 'w') as f:
                f.write(f"PAIRS TRADING ANALYSIS: {name}\n")
                f.write(f"="*50 + "\n\n")
                f.write(f"Stocks: {syms[0]} vs {syms[1]}\n")
                f.write(f"Data points: {len(df_clean)}\n")
                f.write(f"Test period: {len(test_data)} days\n\n")
                
                f.write(f"COINTEGRATION TEST:\n")
                f.write(f"  Test Statistic: {score:.4f}\n")
                f.write(f"  P-value: {p_val:.4f}\n")
                f.write(f"  Cointegrated: {is_cointegrated}\n\n")
                
                f.write(f"STATISTICAL BASELINE:\n")
                f.write(f"  Accuracy:  {stat_accuracy:.2%}\n")
                f.write(f"  Precision: {stat_precision:.2%}\n")
                f.write(f"  F1-Score:  {stat_f1:.2%}\n\n")
                
                f.write(f"XGBOOST ENHANCED:\n")
                f.write(f"  Accuracy:  {ml_accuracy:.2%}\n")
                f.write(f"  Precision: {ml_precision:.2%}\n")
                f.write(f"  F1-Score:  {ml_f1:.2%}\n\n")
                
                f.write(f"IMPROVEMENT: {improvement:+.2f}%\n")
                f.write(f"\nCONCLUSION:\n")
                f.write(f"XGBoost {'ENHANCES' if improvement > 0 else 'DOES NOT ENHANCE'} ")
                f.write(f"the predictive power of the statistical cointegration strategy.\n")
            
            # 8. Save detailed data
            test_data_export = test_data.copy()
            test_data_export['ML_Prediction'] = y_pred_ml
            test_data_export['Stat_Prediction'] = y_pred_stat
            test_data_export.to_csv(f"results/data/{name}_detailed_results.csv")
            
            # 9. Create visualization
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot 1: Z-Score with signals
            axes[0].plot(df_clean.index, df_clean['Z_Score'], label='Z-Score', color='blue', alpha=0.7)
            axes[0].axhline(2, color='red', ls='--', label='Sell Threshold (+2)', alpha=0.7)
            axes[0].axhline(-2, color='green', ls='--', label='Buy Threshold (-2)', alpha=0.7)
            axes[0].axhline(0, color='black', ls='-', alpha=0.3)
            axes[0].set_title(f"{name} Pair ({syms[0]}/{syms[1]}): Mean Reversion Signal", fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Z-Score', fontsize=11)
            axes[0].legend(loc='upper right')
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Performance comparison
            metrics = ['Accuracy', 'Precision', 'F1-Score']
            stat_scores = [stat_accuracy, stat_precision, stat_f1]
            ml_scores = [ml_accuracy, ml_precision, ml_f1]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = axes[1].bar(x - width/2, stat_scores, width, label='Statistical Baseline', color='#FF6B6B', alpha=0.8)
            bars2 = axes[1].bar(x + width/2, ml_scores, width, label='XGBoost Enhanced', color='#4ECDC4', alpha=0.8)
            
            axes[1].set_ylabel('Score', fontsize=11)
            axes[1].set_title(f'Performance Comparison: Statistical vs ML-Enhanced\nImprovement: {improvement:+.1f}%', 
                            fontsize=14, fontweight='bold')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(metrics, fontsize=11)
            axes[1].legend(loc='lower right')
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].set_ylim(0, 1)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1%}',
                               ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f"results/plots/{name}_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n✓ Saved comparison plot: results/plots/{name}_comparison.png")
            
        except Exception as e:
            print(f"\n❌ ERROR processing {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save overall summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv("results/data/overall_summary.csv", index=False)
        
        print(f"\n\n{'='*60}")
        print("OVERALL SUMMARY")
        print('='*60)
        print(summary_df.to_string(index=False))
        print(f"\n✓ Saved to results/data/overall_summary.csv")
        
        # Calculate average improvement
        avg_improvement = summary_df['Improvement_Percent'].mean()
        print(f"\n{'='*60}")
        print(f"AVERAGE IMPROVEMENT ACROSS ALL PAIRS: {avg_improvement:+.2f}%")
        print(f"{'='*60}")

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("SCRIPT STARTED")
        print("=" * 60)
        run()
        print("\n" + "=" * 60)
        print("SCRIPT COMPLETED SUCCESSFULLY")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print("FATAL ERROR:")
        print(str(e))
        print("=" * 60)
        import traceback
        traceback.print_exc()
        raise
