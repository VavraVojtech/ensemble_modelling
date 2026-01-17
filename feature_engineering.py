import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import holidays
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'input_folder': 'input_data',
    'output_folder': 'output_features',
    'target_col': 'spotreba_cr',
    'seed': 42
}

os.makedirs(CONFIG['output_folder'], exist_ok=True)

# ==========================================
# 2. FEATURE ENGINEERING (Consistent with Master Script)
# ==========================================
def calculate_ote_effective_temperature(temps, alpha=0.5):
    """Vypočte efektivní teplotu dle metodiky OTE (vyhlazená řada)."""
    t_eff = np.zeros_like(temps)
    t_eff[0] = temps[0]
    for t in range(1, len(temps)):
        t_eff[t] = alpha * temps[t] + (1 - alpha) * t_eff[t-1]
    return t_eff

def prepare_dataset():
    print("--- Loading Data ---")
    path_ote = os.path.join(CONFIG['input_folder'], 'OTE_NG_ODCHYLKY_NC_BAL.csv')
    path_weather = os.path.join(CONFIG['input_folder'], 'OpenMeteo_train_data_6_6.csv')
    
    if not os.path.exists(path_ote) or not os.path.exists(path_weather):
        raise FileNotFoundError(f"Missing input files in {CONFIG['input_folder']}")

    # OTE Data
    df_ote = pd.read_csv(path_ote)
    df_ote['DATE'] = pd.to_datetime(df_ote['DATE'])
    df_ote = df_ote[['DATE', CONFIG['target_col']]].copy()
    # Plyn se spotřebovává, takže v datech je často záporné znaménko. Otočíme na kladné.
    df_ote[CONFIG['target_col']] = df_ote[CONFIG['target_col']] * (-1)
    df_ote.rename(columns={'DATE': 'date'}, inplace=True)

    # Weather Data
    df_weather = pd.read_csv(path_weather)
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    if 'visibility' in df_weather.columns: 
        df_weather.drop(columns=['visibility'], inplace=True)

    # Merge
    df = pd.merge(df_weather, df_ote, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)

    # --- Feature Engineering ---
    print("--- Generating Features ---")
    
    # 1. Effective Temperature (OTE method)
    df['temp_effective'] = calculate_ote_effective_temperature(df['temperature_2m'].values)
    
    # 2. Calendar Features
    cz_holidays = holidays.CZ()
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in cz_holidays else 0)
    # is_working_day: není víkend A není svátek
    df['is_working_day'] = ((df['is_weekend'] == 0) & (df['is_holiday'] == 0)).astype(int)
    
    # Cyclical encoding of day of year
    df['day_of_year'] = df['date'].dt.dayofyear
    df['sin_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

    # 3. Lags & Rolling Windows (Weather)
    weather_cols = ['temperature_2m', 'apparent_temperature', 'wind_speed_10m', 'temp_effective']
    for col in weather_cols:
        # Lags
        for lag in [1, 2]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        # Rolling Means
        for w in [3, 7]:
            df[f'{col}_roll_{w}'] = df[col].rolling(window=w).mean()

    # 4. Target Lags (Autoregression)
    # Pozor: Při reálné predikci musíme mít tyto hodnoty k dispozici.
    # Lag 1 je včerejší spotřeba (známe ji dnes ráno?). Obvykle ano.
    for lag in [1, 2, 7, 14]:
        df[f'target_lag_{lag}'] = df[CONFIG['target_col']].shift(lag)

    # Cleanup NaN (due to lags)
    df.dropna(inplace=True)
    return df.reset_index(drop=True)

# ==========================================
# 3. ANALYSIS
# ==========================================
def plot_importance(importance_df, title, filename):
    plt.figure(figsize=(10, 8))
    # Seřadit a vzít top 20
    top_df = importance_df.sort_values(by='Importance', ascending=True).tail(20)
    
    # Barva podle kladné/záporné hodnoty (pro Ridge)
    colors = ['red' if x < 0 else 'blue' for x in top_df['Importance']]
    if 'AbsImportance' in top_df.columns: # Pro Tree model barva uniformní
        colors = 'skyblue'
        
    plt.barh(top_df['Feature'], top_df['Importance'], color=colors)
    plt.title(title, fontsize=14)
    plt.xlabel('Importance Score')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_folder'], filename), dpi=300)
    plt.close()
    print(f"Graph saved: {filename}")

def main():
    df = prepare_dataset()
    
    # Prepare X and y
    feature_cols = [c for c in df.columns if c not in ['date', CONFIG['target_col']]]
    X = df[feature_cols]
    y = df[CONFIG['target_col']]
    
    # Split (Time-based split, just for validation)
    # Pro Feature Importance je lepší použít co nejvíce dat
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # --- A) Tree Based Importance (XGBoost) ---
    print("\n--- 1. XGBoost Feature Importance ---")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=CONFIG['seed'],
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # Get Importance (Gain is usually the best metric for "information value")
    importance = xgb_model.get_booster().get_score(importance_type='gain')
    # Map features names correctly (get_score returns f0, f1... sometimes)
    # Sklearn API makes it easier:
    imp_vals = xgb_model.feature_importances_
    
    df_xgb_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': imp_vals,
        'AbsImportance': np.abs(imp_vals) # Just for sorting
    })
    
    print("Top 10 Features (XGBoost):")
    print(df_xgb_imp.sort_values('Importance', ascending=False).head(10))
    plot_importance(df_xgb_imp, 'XGBoost Feature Importance (Gain)', 'feature_importance_xgboost.png')

    # --- B) Linear Model Coefficients (Ridge) ---
    print("\n--- 2. Ridge Regression Coefficients ---")
    # Linear models require scaling!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    
    coeffs = ridge_model.coef_
    
    df_ridge_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': coeffs,
        'AbsImportance': np.abs(coeffs) # For sorting visualization
    })
    
    # Sort by Absolute value but show Real value (to see direction)
    df_ridge_imp_sorted = df_ridge_imp.sort_values('AbsImportance', ascending=False)
    
    print("Top 10 Features (Ridge - Absolute Magnitude):")
    print(df_ridge_imp_sorted.head(10))
    
    # Plotting: We sort by pure value for bar chart logic, or abs value? 
    # Usually better to see the strongest effects at the bottom.
    # Let's plot the ones with highest ABS importance
    plot_importance(df_ridge_imp.sort_values('AbsImportance'), 
                   'Ridge Regression Coefficients (Standardized)', 
                   'feature_importance_ridge.png')

    # Save data
    df_xgb_imp.sort_values('Importance', ascending=False).to_csv(os.path.join(CONFIG['output_folder'], 'importance_xgboost.csv'), index=False)
    df_ridge_imp_sorted.to_csv(os.path.join(CONFIG['output_folder'], 'importance_ridge.csv'), index=False)
    print("\nAll done. Check 'output_features' folder.")

if __name__ == "__main__":
    main()