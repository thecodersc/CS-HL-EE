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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yfinance as yf
from datetime import datetime, timedelta
import warnings

# Visual setup
plt.style.use('ggplot')
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================================================
# -----------------------------
# CONFIGURATION
# ============================================================================
# -----------------------------
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/data", exist_ok=True)

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
PAIRS = {"Energy": ["XOM", "CVX"], "Tech": ["NVDA", "AMD"]}

# Create directories
for directory in [Config.PLOT_DIR, Config.DATA_DIR, Config.REPORT_DIR]:
    os.makedirs(directory, exist_ok=True)
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=730)

# ============================================================================
# DATA ACQUISITION MODULE
# ============================================================================

class DataFetcher:
    """Handles data download from Yahoo Finance"""
# -----------------------------
# FUNCTION TO PULL DATA
# -----------------------------
def fetch_stock_data(symbol: str) -> pd.DataFrame:
    print(f"  Downloading {symbol} data from Yahoo Finance...")

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
        
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end)
        df = ticker.history(start=START_DATE, end=END_DATE)

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
            raise Exception(f"No data returned for {symbol}")

        df = df[['Close']].copy()
        df.index.name = 'Date'

        elapsed = time.time() - start_time
        print(f"✓ {len(df)} days ({elapsed:.2f}s)")
        
        print(f"  ✓ Downloaded {len(df)} trading days")
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
    except Exception as e:
        raise Exception(f"Error fetching {symbol}: {str(e)}")

# ============================================================================
# BASELINE STRATEGY MODULE
# ============================================================================

class BaselineStrategy:
    """Traditional statistical pairs trading strategy"""
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
    # Volatility
    df['Z_Volatility'] = df['Z_Score'].rolling(10).std()

    def __init__(self):
        self.model = None
        self.feature_importance = None
    # Momentum indicators
    df['Z_Momentum'] = df['Z_Score'] - df['Z_Score'].shift(5)
    df['Z_ROC'] = df['Z_Score'].pct_change(5)  # Rate of change

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
    # Mean reversion indicators
    df['Distance_MA20'] = df['Z_Score'] - df['Z_MA_20']
    df['Crossover_MA5_MA20'] = (df['Z_MA_5'] > df['Z_MA_20']).astype(int)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        return self.model.predict(X_test)
    # Extreme values
    df['Extreme_High'] = (df['Z_Score'] > 2).astype(int)
    df['Extreme_Low'] = (df['Z_Score'] < -2).astype(int)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance DataFrame"""
        return self.feature_importance
    return df

# ============================================================================
# EVALUATION MODULE
# ============================================================================

class PerformanceEvaluator:
    """Evaluates and compares model performance"""
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
    # Signals: Trade when Z-score is extreme relative to recent history
    df['Stat_Signal'] = 0

    @staticmethod
    def print_metrics(metrics: dict):
        """Pretty print evaluation metrics"""
        print(f"\n    {metrics['Model']}:")
        print(f"      Accuracy:  {metrics['Accuracy']:.2%}")
        print(f"      Precision: {metrics['Precision']:.2%}")
        print(f"      Recall:    {metrics['Recall']:.2%}")
        print(f"      F1-Score:  {metrics['F1_Score']:.2%}")
    # More sophisticated: use percentiles for entry
    upper_threshold = df['Z_Score'].rolling(60).quantile(0.85)
    lower_threshold = df['Z_Score'].rolling(60).quantile(0.15)

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
    df.loc[df['Z_Score'] < lower_threshold, 'Stat_Signal'] = 1  # Buy
    df.loc[df['Z_Score'] > upper_threshold, 'Stat_Signal'] = -1  # Sell

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
    # Target: Will spread mean-revert in next 3 days?
    df['Stat_Target'] = np.where(
        np.abs(df['Z_Score'].shift(-3)) < np.abs(df['Z_Score']), 1, 0
    )

    print("="*80)
    print("CS HL EXTENDED ESSAY: ML ENHANCEMENT OF PAIRS TRADING")
    print("="*80)
    print(f"\nResearch Question: To what extent does XGBoost enhance")
    print(f"cointegration-based pairs trading?\n")
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
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print("FATAL ERROR:")
        print(str(e))
        print("=" * 60)
        import traceback
