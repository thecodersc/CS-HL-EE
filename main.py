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
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/data", exist_ok=True)

PAIRS = {"Energy": ["XOM", "CVX"], "Tech": ["NVDA", "AMD"]}

END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=730)

# -----------------------------
# FUNCTION TO PULL DATA
# -----------------------------
def fetch_stock_data(symbol: str) -> pd.DataFrame:
    print(f"  Downloading {symbol} data from Yahoo Finance...")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=START_DATE, end=END_DATE)
        
        if df.empty:
            raise Exception(f"No data returned for {symbol}")
        
        df = df[['Close']].copy()
        df.index.name = 'Date'
        
        print(f"  ✓ Downloaded {len(df)} trading days")
        return df
        
    except Exception as e:
        raise Exception(f"Error fetching {symbol}: {str(e)}")

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def create_features(df):
    """
    Create advanced features for ML model
    """
    # Rolling statistics
    df['Z_MA_5'] = df['Z_Score'].rolling(5).mean()
    df['Z_MA_10'] = df['Z_Score'].rolling(10).mean()
    df['Z_MA_20'] = df['Z_Score'].rolling(20).mean()
    
    # Volatility
    df['Z_Volatility'] = df['Z_Score'].rolling(10).std()
    
    # Momentum indicators
    df['Z_Momentum'] = df['Z_Score'] - df['Z_Score'].shift(5)
    df['Z_ROC'] = df['Z_Score'].pct_change(5)  # Rate of change
    
    # Mean reversion indicators
    df['Distance_MA20'] = df['Z_Score'] - df['Z_MA_20']
    df['Crossover_MA5_MA20'] = (df['Z_MA_5'] > df['Z_MA_20']).astype(int)
    
    # Extreme values
    df['Extreme_High'] = (df['Z_Score'] > 2).astype(int)
    df['Extreme_Low'] = (df['Z_Score'] < -2).astype(int)
    
    return df

# -----------------------------
# IMPROVED BASELINE STRATEGY
# -----------------------------
def statistical_baseline_strategy(df):
    """
    Improved statistical strategy with dynamic thresholds
    """
    # Use rolling mean and std for adaptive thresholds
    rolling_mean = df['Z_Score'].rolling(20).mean()
    rolling_std = df['Z_Score'].rolling(20).std()
    
    # Signals: Trade when Z-score is extreme relative to recent history
    df['Stat_Signal'] = 0
    
    # More sophisticated: use percentiles for entry
    upper_threshold = df['Z_Score'].rolling(60).quantile(0.85)
    lower_threshold = df['Z_Score'].rolling(60).quantile(0.15)
    
    df.loc[df['Z_Score'] < lower_threshold, 'Stat_Signal'] = 1  # Buy
    df.loc[df['Z_Score'] > upper_threshold, 'Stat_Signal'] = -1  # Sell
    
    # Target: Will spread mean-revert in next 3 days?
    df['Stat_Target'] = np.where(
        np.abs(df['Z_Score'].shift(-3)) < np.abs(df['Z_Score']), 1, 0
    )
    
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
            
            # 5. Feature Engineering
            df = create_features(df)
            
            # 6. Create better target: Profitable mean reversion
            # Target = 1 if entering now would be profitable in 3-5 days
            df['ML_Target'] = np.where(
                (np.abs(df['Z_Score'].shift(-3)) < np.abs(df['Z_Score'])) & 
                (np.abs(df['Z_Score'].shift(-5)) < np.abs(df['Z_Score'])), 
                1, 0
            )
            
            # Drop NaN rows created by feature engineering
            df_clean = df.dropna()
            
            # Split data: 80% train, 20% test
            split_idx = int(len(df_clean) * 0.8)
            train_data = df_clean.iloc[:split_idx]
            test_data = df_clean.iloc[split_idx:]
            
            print(f"\nDataset Split:")
            print(f"  Training: {len(train_data)} days")
            print(f"  Testing: {len(test_data)} days")
            
            # 7. Train XGBoost with MANY features
            feature_cols = ['Z_Score', 'Z_MA_5', 'Z_MA_10', 'Z_MA_20', 
                          'Z_Volatility', 'Z_Momentum', 'Z_ROC',
                          'Distance_MA20', 'Crossover_MA5_MA20',
                          'Extreme_High', 'Extreme_Low']
            
            X_train = train_data[feature_cols]
            y_train = train_data['ML_Target']
            X_test = test_data[feature_cols]
            y_test = test_data['ML_Target']
            
            # Check class balance
            class_balance = y_train.value_counts()
            print(f"\nTraining Set Class Balance:")
            print(f"  Class 0 (No Trade): {class_balance[0]}")
            print(f"  Class 1 (Trade): {class_balance[1]}")
            
            # Calculate scale_pos_weight for imbalanced classes
            scale_pos_weight = class_balance[0] / class_balance[1]
            
            model = xgb.XGBClassifier(
                eval_metric='logloss',
                max_depth=4,
                learning_rate=0.05,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                scale_pos_weight=scale_pos_weight  # Handle class imbalance
            )
            model.fit(X_train, y_train)
            
            # 8. Predictions
            y_pred_ml = model.predict(X_test)
            
            # Statistical baseline predictions on test set
            y_pred_stat = test_data['Stat_Signal'].apply(lambda x: 1 if x != 0 else 0)
            
            # 9. COMPARISON METRICS
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
            print(f"\nSTATISTICAL BASELINE (Adaptive Thresholds):")
            print(f"  Accuracy:  {stat_accuracy:.2%}")
            print(f"  Precision: {stat_precision:.2%}")
            print(f"  Recall:    {stat_recall:.2%}")
            print(f"  F1-Score:  {stat_f1:.2%}")
            
            print(f"\nXGBOOST ENHANCED MODEL (11 features):")
            print(f"  Accuracy:  {ml_accuracy:.2%}")
            print(f"  Precision: {ml_precision:.2%}")
            print(f"  Recall:    {ml_recall:.2%}")
            print(f"  F1-Score:  {ml_f1:.2%}")
            
            print(f"\n{'='*60}")
            print(f"IMPROVEMENT: {improvement:+.2f}%")
            print(f"CONCLUSION: XGBoost {'ENHANCES' if improvement > 0 else 'DOES NOT ENHANCE'} the strategy")
            print(f"{'='*60}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print(f"\nTop 5 Most Important Features:")
            for idx, row in feature_importance.head(5).iterrows():
                print(f"  {row['Feature']}: {row['Importance']:.4f}")
            
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
                'Improvement_Percent': improvement,
                'Top_Feature': feature_importance.iloc[0]['Feature']
            }
            all_results.append(results)
            
            # 10. Save summary to file
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
                
                f.write(f"XGBOOST ENHANCED (11 features):\n")
                f.write(f"  Accuracy:  {ml_accuracy:.2%}\n")
                f.write(f"  Precision: {ml_precision:.2%}\n")
                f.write(f"  F1-Score:  {ml_f1:.2%}\n\n")
                
                f.write(f"IMPROVEMENT: {improvement:+.2f}%\n\n")
                
                f.write(f"FEATURE IMPORTANCE:\n")
                for idx, row in feature_importance.iterrows():
                    f.write(f"  {row['Feature']}: {row['Importance']:.4f}\n")
                
                f.write(f"\nCONCLUSION:\n")
                f.write(f"XGBoost {'ENHANCES' if improvement > 0 else 'DOES NOT ENHANCE'} ")
                f.write(f"the predictive power of the statistical cointegration strategy.\n")
                f.write(f"The most important feature for prediction is: {feature_importance.iloc[0]['Feature']}\n")
            
            # 11. Save detailed data
            test_data_export = test_data.copy()
            test_data_export['ML_Prediction'] = y_pred_ml
            test_data_export['Stat_Prediction'] = y_pred_stat
            test_data_export.to_csv(f"results/data/{name}_detailed_results.csv")
            
            # Save feature importance
            feature_importance.to_csv(f"results/data/{name}_feature_importance.csv", index=False)
            
            # 12. Create visualization
            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            # Plot 1: Z-Score with signals
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(df_clean.index, df_clean['Z_Score'], label='Z-Score', color='blue', alpha=0.7, linewidth=1)
            ax1.axhline(2, color='red', ls='--', label='Traditional Threshold (+2)', alpha=0.5)
            ax1.axhline(-2, color='green', ls='--', label='Traditional Threshold (-2)', alpha=0.5)
            ax1.axhline(0, color='black', ls='-', alpha=0.3)
            ax1.set_title(f"{name} Pair ({syms[0]}/{syms[1]}): Mean Reversion Signal", fontsize=14, fontweight='bold')
            ax1.set_ylabel('Z-Score', fontsize=11)
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Performance comparison
            ax2 = fig.add_subplot(gs[1, 0])
            metrics = ['Accuracy', 'Precision', 'F1-Score']
            stat_scores = [stat_accuracy, stat_precision, stat_f1]
            ml_scores = [ml_accuracy, ml_precision, ml_f1]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, stat_scores, width, label='Statistical', color='#FF6B6B', alpha=0.8)
            bars2 = ax2.bar(x + width/2, ml_scores, width, label='XGBoost', color='#4ECDC4', alpha=0.8)
            
            ax2.set_ylabel('Score', fontsize=11)
            ax2.set_title(f'Performance Comparison\nImprovement: {improvement:+.1f}%', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(metrics, fontsize=10)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_ylim(0, 1)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1%}', ha='center', va='bottom', fontsize=8)
            
            # Plot 3: Feature Importance
            ax3 = fig.add_subplot(gs[1, 1])
            top_features = feature_importance.head(8)
            ax3.barh(range(len(top_features)), top_features['Importance'], color='#95E1D3')
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features['Feature'], fontsize=9)
            ax3.set_xlabel('Importance', fontsize=10)
            ax3.set_title('Top 8 Feature Importance', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')
            
            # Plot 4: Prediction Distribution
            ax4 = fig.add_subplot(gs[2, :])
            test_dates = test_data.index
            ax4.scatter(test_dates[y_pred_ml == 1], [1]*sum(y_pred_ml), 
                       label='ML Predicted Trade', color='#4ECDC4', alpha=0.6, s=20)
            ax4.scatter(test_dates[y_pred_stat == 1], [0]*sum(y_pred_stat), 
                       label='Statistical Predicted Trade', color='#FF6B6B', alpha=0.6, s=20)
            ax4.scatter(test_dates[y_test == 1], [-1]*sum(y_test), 
                       label='Actual Profitable', color='gold', alpha=0.8, s=30, marker='*')
            ax4.set_yticks([1, 0, -1])
            ax4.set_yticklabels(['ML', 'Statistical', 'Actual'])
            ax4.set_title('Trading Signals Over Time (Test Period)', fontsize=12, fontweight='bold')
            ax4.legend(loc='upper right', fontsize=9)
            ax4.grid(True, alpha=0.3)
            
            plt.savefig(f"results/plots/{name}_comprehensive_analysis.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n✓ Saved comprehensive plot: results/plots/{name}_comprehensive_analysis.png")
            
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
