import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from statsmodels.tsa.stattools import coint
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for the analysis"""
    # Data parameters
    PAIRS = {
        "Energy": ["XOM", "CVX"], 
        "Tech": ["NVDA", "AMD"]
    }
    
    END_DATE = datetime.today()
    START_DATE = END_DATE - timedelta(days=730)
    
    # Model parameters
    TRAIN_TEST_SPLIT = 0.8
    RANDOM_STATE = 42
    
    # Feature engineering windows
    MA_SHORT = 5
    MA_MEDIUM = 10
    MA_LONG = 20
    VOLATILITY_WINDOW = 10
    MOMENTUM_WINDOW = 5
    
    # Baseline strategy parameters
    PERCENTILE_WINDOW = 60
    UPPER_PERCENTILE = 0.85
    LOWER_PERCENTILE = 0.15
    
    # XGBoost hyperparameters
    XGBOOST_PARAMS = {
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'random_state': RANDOM_STATE
    }
    
    # Output directories
    OUTPUT_DIR = "results/cs_ee"
    PLOT_DIR = f"{OUTPUT_DIR}/plots"
    DATA_DIR = f"{OUTPUT_DIR}/data"
    REPORT_DIR = f"{OUTPUT_DIR}/reports"

# Create directories
for directory in [Config.PLOT_DIR, Config.DATA_DIR, Config.REPORT_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DATA ACQUISITION MODULE
# ============================================================================

class DataFetcher:
    """Handles data download from Yahoo Finance"""
    
    @staticmethod
    def fetch_stock_data(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance
        
        Args:
            symbol: Stock ticker symbol
            start: Start date
            end: End date
            
        Returns:
            DataFrame with Close prices
        """
        print(f"    Fetching {symbol}...", end=" ")
        start_time = time.time()
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end)
        
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        df = df[['Close']].copy()
        df.index.name = 'Date'
        
        elapsed = time.time() - start_time
        print(f"✓ {len(df)} days ({elapsed:.2f}s)")
        
        return df

# ============================================================================
# FEATURE ENGINEERING MODULE
# ============================================================================

class FeatureEngineer:
    """Creates features for machine learning model"""
    
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate 11 features from Z-score time series
        
        Features:
        1. Z_Score - Raw standardized spread
        2-4. Z_MA_5, Z_MA_10, Z_MA_20 - Moving averages
        5. Z_Volatility - Rolling standard deviation
        6. Z_Momentum - Change over window
        7. Z_ROC - Rate of change (percentage)
        8. Distance_MA20 - Distance from long-term MA
        9. Crossover_MA5_MA20 - Trend crossover signal
        10. Extreme_High - Above +2σ threshold
        11. Extreme_Low - Below -2σ threshold
        """
        df = df.copy()
        
        # Moving averages
        df['Z_MA_5'] = df['Z_Score'].rolling(Config.MA_SHORT).mean()
        df['Z_MA_10'] = df['Z_Score'].rolling(Config.MA_MEDIUM).mean()
        df['Z_MA_20'] = df['Z_Score'].rolling(Config.MA_LONG).mean()
        
        # Volatility
        df['Z_Volatility'] = df['Z_Score'].rolling(Config.VOLATILITY_WINDOW).std()
        
        # Momentum indicators
        df['Z_Momentum'] = df['Z_Score'] - df['Z_Score'].shift(Config.MOMENTUM_WINDOW)
        df['Z_ROC'] = df['Z_Score'].pct_change(Config.MOMENTUM_WINDOW)
        
        # Mean reversion indicators
        df['Distance_MA20'] = df['Z_Score'] - df['Z_MA_20']
        df['Crossover_MA5_MA20'] = (df['Z_MA_5'] > df['Z_MA_20']).astype(int)
        
        # Extreme values (binary features)
        df['Extreme_High'] = (df['Z_Score'] > 2).astype(int)
        df['Extreme_Low'] = (df['Z_Score'] < -2).astype(int)
        
        return df
    
    @staticmethod
    def get_feature_names() -> list:
        """Return list of feature column names"""
        return ['Z_Score', 'Z_MA_5', 'Z_MA_10', 'Z_MA_20', 
                'Z_Volatility', 'Z_Momentum', 'Z_ROC',
                'Distance_MA20', 'Crossover_MA5_MA20',
                'Extreme_High', 'Extreme_Low']

# ============================================================================
# BASELINE STRATEGY MODULE
# ============================================================================

class BaselineStrategy:
    """Traditional statistical pairs trading strategy"""
    
    @staticmethod
    def apply(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply adaptive percentile-based trading strategy
        
        Algorithm:
        1. Calculate rolling percentiles (60-day window)
        2. Generate signals when Z-score exceeds thresholds
        3. Create target: profitable mean reversion in 3-5 days
        """
        df = df.copy()
        
        # Adaptive thresholds based on rolling percentiles
        upper_threshold = df['Z_Score'].rolling(Config.PERCENTILE_WINDOW).quantile(Config.UPPER_PERCENTILE)
        lower_threshold = df['Z_Score'].rolling(Config.PERCENTILE_WINDOW).quantile(Config.LOWER_PERCENTILE)
        
        # Generate trading signals
        df['Stat_Signal'] = 0
        df.loc[df['Z_Score'] < lower_threshold, 'Stat_Signal'] = 1  # Buy signal
        df.loc[df['Z_Score'] > upper_threshold, 'Stat_Signal'] = -1  # Sell signal
        
        # Target: Mean reversion occurs in next 3-5 days
        df['Stat_Target'] = np.where(
            (np.abs(df['Z_Score'].shift(-3)) < np.abs(df['Z_Score'])) & 
            (np.abs(df['Z_Score'].shift(-5)) < np.abs(df['Z_Score'])), 
            1, 0
        )
        
        return df

# ============================================================================
# MACHINE LEARNING MODULE
# ============================================================================

class MLModel:
    """XGBoost-based enhanced trading model"""
    
    def __init__(self):
        self.model = None
        self.feature_importance = None
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              scale_pos_weight: float = None):
        """
        Train XGBoost classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            scale_pos_weight: Weight for positive class (handles imbalance)
        """
        params = Config.XGBOOST_PARAMS.copy()
        
        if scale_pos_weight:
            params['scale_pos_weight'] = scale_pos_weight
        
        self.model = xgb.XGBClassifier(**params)
        
        print("    Training XGBoost model...")
        start_time = time.time()
        self.model.fit(X_train, y_train, verbose=False)
        elapsed = time.time() - start_time
        
        print(f"    ✓ Training complete ({elapsed:.2f}s)")
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        return self.model.predict(X_test)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance DataFrame"""
        return self.feature_importance

# ============================================================================
# EVALUATION MODULE
# ============================================================================

class PerformanceEvaluator:
    """Evaluates and compares model performance"""
    
    @staticmethod
    def evaluate(y_true, y_pred, model_name: str) -> dict:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary with accuracy, precision, recall, F1-score
        """
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1_Score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: dict):
        """Pretty print evaluation metrics"""
        print(f"\n    {metrics['Model']}:")
        print(f"      Accuracy:  {metrics['Accuracy']:.2%}")
        print(f"      Precision: {metrics['Precision']:.2%}")
        print(f"      Recall:    {metrics['Recall']:.2%}")
        print(f"      F1-Score:  {metrics['F1_Score']:.2%}")
    
    @staticmethod
    def calculate_improvement(baseline_acc: float, ml_acc: float) -> float:
        """Calculate percentage improvement"""
        if baseline_acc == 0:
            return 0.0
        return ((ml_acc - baseline_acc) / baseline_acc) * 100

# ============================================================================
# VISUALIZATION MODULE
# ============================================================================

class Visualizer:
    """Creates comprehensive visualizations"""
    
    @staticmethod
    def create_comprehensive_plot(pair_name: str, syms: list, df: pd.DataFrame,
                                 stat_metrics: dict, ml_metrics: dict,
                                 feature_importance: pd.DataFrame,
                                 y_test: pd.Series, y_pred_stat: np.ndarray,
                                 y_pred_ml: np.ndarray, test_data: pd.DataFrame):
        """
        Create comprehensive 4-panel visualization
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        improvement = PerformanceEvaluator.calculate_improvement(
            stat_metrics['Accuracy'], ml_metrics['Accuracy']
        )
        
        fig.suptitle(
            f'CS HL EE: ML Enhancement Analysis - {pair_name} Pair ({syms[0]}/{syms[1]})\n'
            f'Improvement: {improvement:+.1f}%',
            fontsize=16, fontweight='bold'
        )
        
        # Plot 1: Z-Score Time Series
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df.index, df['Z_Score'], label='Z-Score', color='#2E86AB', 
                linewidth=1.5, alpha=0.8)
        ax1.axhline(2, color='#E63946', ls='--', label='Threshold +2σ', alpha=0.6)
        ax1.axhline(-2, color='#06A77D', ls='--', label='Threshold -2σ', alpha=0.6)
        ax1.axhline(0, color='black', ls='-', alpha=0.3, linewidth=0.8)
        ax1.fill_between(df.index, -2, 2, alpha=0.1, color='#06A77D')
        ax1.set_title('Mean Reversion Signal (Z-Score)', fontsize=14, pad=10)
        ax1.set_ylabel('Z-Score', fontsize=11)
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confusion Matrices
        ax2 = fig.add_subplot(gs[1, 0])
        cm_stat = confusion_matrix(y_test, y_pred_stat)
        sns.heatmap(cm_stat, annot=True, fmt='d', cmap='Blues', ax=ax2,
                   cbar_kws={'label': 'Count'})
        ax2.set_title(f'Statistical Baseline\nAccuracy: {stat_metrics["Accuracy"]:.1%}',
                     fontsize=12, pad=10)
        ax2.set_ylabel('Actual', fontsize=10)
        ax2.set_xlabel('Predicted', fontsize=10)
        
        ax3 = fig.add_subplot(gs[1, 1])
        cm_ml = confusion_matrix(y_test, y_pred_ml)
        sns.heatmap(cm_ml, annot=True, fmt='d', cmap='Greens', ax=ax3,
                   cbar_kws={'label': 'Count'})
        ax3.set_title(f'XGBoost Enhanced\nAccuracy: {ml_metrics["Accuracy"]:.1%}',
                     fontsize=12, pad=10)
        ax3.set_ylabel('Actual', fontsize=10)
        ax3.set_xlabel('Predicted', fontsize=10)
        
        # Plot 3: Performance Comparison
        ax4 = fig.add_subplot(gs[2, 0])
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        stat_scores = [stat_metrics['Accuracy'], stat_metrics['Precision'],
                      stat_metrics['Recall'], stat_metrics['F1_Score']]
        ml_scores = [ml_metrics['Accuracy'], ml_metrics['Precision'],
                    ml_metrics['Recall'], ml_metrics['F1_Score']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, stat_scores, width, label='Statistical',
                       color='#E76F51', alpha=0.8)
        bars2 = ax4.bar(x + width/2, ml_scores, width, label='XGBoost',
                       color='#2A9D8F', alpha=0.8)
        
        ax4.set_ylabel('Score', fontsize=11)
        ax4.set_title('Performance Metrics Comparison', fontsize=12, pad=10)
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics, fontsize=10)
        ax4.legend(framealpha=0.9)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1%}', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Feature Importance
        ax5 = fig.add_subplot(gs[2, 1])
        top_features = feature_importance.head(8)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        ax5.barh(range(len(top_features)), top_features['Importance'], color=colors)
        ax5.set_yticks(range(len(top_features)))
        ax5.set_yticklabels(top_features['Feature'], fontsize=10)
        ax5.set_xlabel('Importance Score', fontsize=10)
        ax5.set_title('Top 8 Feature Importance (XGBoost)', fontsize=12, pad=10)
        ax5.grid(True, alpha=0.3, axis='x')
        ax5.invert_yaxis()
        
        plt.savefig(f"{Config.PLOT_DIR}/{pair_name}_ml_analysis.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def run_analysis():
    """Main analysis pipeline"""
    
    print("="*80)
    print("CS HL EXTENDED ESSAY: ML ENHANCEMENT OF PAIRS TRADING")
    print("="*80)
    print(f"\nResearch Question: To what extent does XGBoost enhance")
    print(f"cointegration-based pairs trading?\n")
    
    all_results = []
    
    for pair_name, syms in Config.PAIRS.items():
        print(f"\n{'='*80}")
        print(f"ANALYZING {pair_name.upper()} PAIR: {syms[0]} vs {syms[1]}")
        print('='*80)
        
        # ====================================================================
        # PHASE 1: DATA COLLECTION
        # ====================================================================
        print("\n[1] DATA COLLECTION")
        df_a = DataFetcher.fetch_stock_data(syms[0], Config.START_DATE, Config.END_DATE)
        df_b = DataFetcher.fetch_stock_data(syms[1], Config.START_DATE, Config.END_DATE)
        
        df = pd.DataFrame({'A': df_a['Close'], 'B': df_b['Close']}).dropna()
        print(f"    ✓ Combined dataset: {len(df)} trading days")
        
        # ====================================================================
        # PHASE 2: COINTEGRATION TEST
        # ====================================================================
        print("\n[2] COINTEGRATION ANALYSIS")
        score, p_val, crit_vals = coint(df['A'], df['B'])
        is_cointegrated = p_val < 0.05
        
        print(f"    Engle-Granger test statistic: {score:.4f}")
        print(f"    P-value: {p_val:.4f}")
        print(f"    Result: {'COINTEGRATED' if is_cointegrated else 'NOT COINTEGRATED'} at 5% level")
        
        # ====================================================================
        # PHASE 3: SPREAD & FEATURE ENGINEERING
        # ====================================================================
        print("\n[3] FEATURE ENGINEERING")
        
        # Calculate spread
        beta = df['A'].mean() / df['B'].mean()
        df['Spread'] = df['A'] - (beta * df['B'])
        df['Z_Score'] = (df['Spread'] - df['Spread'].mean()) / df['Spread'].std()
        
        print(f"    Hedge ratio β = {beta:.4f}")
        print(f"    Z-Score range: [{df['Z_Score'].min():.2f}, {df['Z_Score'].max():.2f}]")
        
        # Apply baseline strategy
        df = BaselineStrategy.apply(df)
        
        # Create ML features
        df = FeatureEngineer.create_features(df)
        print(f"    ✓ Created {len(FeatureEngineer.get_feature_names())} features")
        
        # Create ML target
        df['ML_Target'] = np.where(
            (np.abs(df['Z_Score'].shift(-3)) < np.abs(df['Z_Score'])) & 
            (np.abs(df['Z_Score'].shift(-5)) < np.abs(df['Z_Score'])), 
            1, 0
        )
        
        df_clean = df.dropna()
        
        # ====================================================================
        # PHASE 4: TRAIN/TEST SPLIT
        # ====================================================================
        print("\n[4] DATA SPLITTING")
        split_idx = int(len(df_clean) * Config.TRAIN_TEST_SPLIT)
        train_data = df_clean.iloc[:split_idx]
        test_data = df_clean.iloc[split_idx:]
        
        print(f"    Training set: {len(train_data)} days ({Config.TRAIN_TEST_SPLIT:.0%})")
        print(f"    Test set: {len(test_data)} days ({1-Config.TRAIN_TEST_SPLIT:.0%})")
        
        # Class balance
        class_balance = train_data['ML_Target'].value_counts()
        print(f"    Class balance: {class_balance[1]} positive, {class_balance[0]} negative")
        scale_pos_weight = class_balance[0] / class_balance[1]
        
        # ====================================================================
        # PHASE 5: MODEL TRAINING
        # ====================================================================
        print("\n[5] MODEL TRAINING")
        
        feature_cols = FeatureEngineer.get_feature_names()
        X_train = train_data[feature_cols]
        y_train = train_data['ML_Target']
        X_test = test_data[feature_cols]
        y_test = test_data['ML_Target']
        
        ml_model = MLModel()
        ml_model.train(X_train, y_train, scale_pos_weight)
        
        # ====================================================================
        # PHASE 6: PREDICTION & EVALUATION
        # ====================================================================
        print("\n[6] EVALUATION")
        
        # Predictions
        y_pred_ml = ml_model.predict(X_test)
        y_pred_stat = test_data['Stat_Signal'].apply(lambda x: 1 if x != 0 else 0).values
        
        # Evaluate
        evaluator = PerformanceEvaluator()
        stat_metrics = evaluator.evaluate(y_test,
