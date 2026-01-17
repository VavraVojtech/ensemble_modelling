import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import ElasticNet, BayesianRidge # NOVÉ: BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.compose import TransformedTargetRegressor
from scipy.optimize import nnls, minimize
import joblib
import os
import sys
import warnings
import holidays
from datetime import datetime, timedelta
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# --- Volitelné importy ---
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not installed.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not installed.")

try:
    from neuralprophet import NeuralProphet
    NEURALPROPHET_AVAILABLE = True
except ImportError:
    NEURALPROPHET_AVAILABLE = False
    print("Warning: NeuralProphet not installed.")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMAX_AVAILABLE = True
except ImportError:
    SARIMAX_AVAILABLE = False
    print("Warning: statsmodels not installed.")

try:
    from tbats import TBATS # NOVÉ: TBATS
    TBATS_AVAILABLE = True
except ImportError:
    TBATS_AVAILABLE = False
    print("Warning: tbats not installed (pip install tbats).")

try:
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    import pytorch_lightning as pl
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet, NHiTS
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False
    print("Warning: pytorch-forecasting not installed.")

warnings.filterwarnings('ignore')
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("neuralprophet").setLevel(logging.ERROR)

# ==========================================
# 1. KONFIGURACE
# ==========================================
CONFIG = {
    'input_folder': 'input_data',
    'output_models': 'models',
    'output_graphs': 'output_graph',
    'output_data': 'output_data',
    'split_test_date': '2024-12-01',  
    'split_val_date': '2024-10-01',   
    'target_col': 'spotreba_cr',
    'seed': 42,
    'hyperopt_evals': 100,
    'tft_epochs': 100
}

for folder in [CONFIG['output_models'], CONFIG['output_graphs'], CONFIG['output_data']]:
    os.makedirs(folder, exist_ok=True)

if TFT_AVAILABLE:
    pl.seed_everything(CONFIG['seed'])

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def calculate_ote_effective_temperature(temps, alpha=0.5):
    t_eff = np.zeros_like(temps)
    t_eff[0] = temps[0]
    for t in range(1, len(temps)):
        t_eff[t] = alpha * temps[t] + (1 - alpha) * t_eff[t-1]
    return t_eff

def prepare_dataset():
    print("--- Loading and Engineering Data ---")
    path_ote = os.path.join(CONFIG['input_folder'], 'OTE_NG_ODCHYLKY_NC_BAL.csv')
    path_weather = os.path.join(CONFIG['input_folder'], 'OpenMeteo_train_data_6_6.csv')
    
    if not os.path.exists(path_ote) or not os.path.exists(path_weather):
        raise FileNotFoundError("Input files missing!")

    df_ote = pd.read_csv(path_ote)
    df_ote['DATE'] = pd.to_datetime(df_ote['DATE'])
    df_ote = df_ote[['DATE', CONFIG['target_col']]].copy()
    df_ote[CONFIG['target_col']] = df_ote[CONFIG['target_col']] * (-1)
    df_ote.rename(columns={'DATE': 'date'}, inplace=True)

    df_weather = pd.read_csv(path_weather)
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    if 'visibility' in df_weather.columns: df_weather.drop(columns=['visibility'], inplace=True)

    df = pd.merge(df_weather, df_ote, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)

    # Features
    df['temp_effective'] = calculate_ote_effective_temperature(df['temperature_2m'].values)
    
    cz_holidays = holidays.CZ()
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in cz_holidays else 0)
    df['is_working_day'] = ((df['is_weekend'] == 0) & (df['is_holiday'] == 0)).astype(int)
    
    df['day_of_year'] = df['date'].dt.dayofyear
    df['sin_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

    weather_cols = ['temperature_2m', 'apparent_temperature', 'wind_speed_10m', 'temp_effective']
    for col in weather_cols:
        for lag in [1, 2]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        for w in [3, 7]:
            df[f'{col}_roll_{w}'] = df[col].rolling(window=w).mean()

    for lag in [1, 2, 7, 14]:
        df[f'target_lag_{lag}'] = df[CONFIG['target_col']].shift(lag)

    df.dropna(inplace=True)
    return df.reset_index(drop=True)

# ==========================================
# 3. MODELS
# ==========================================
def get_hyperopt_space(model_name):
    if model_name == 'XGBoost':
        return {'learning_rate': hp.loguniform('lr', np.log(0.01), np.log(0.2)), 'max_depth': hp.quniform('md', 3, 10, 1), 'n_estimators': hp.quniform('ne', 200, 1000, 50)}
    elif model_name == 'LightGBM':
        return {'learning_rate': hp.loguniform('lr', np.log(0.01), np.log(0.2)), 'num_leaves': hp.quniform('nl', 20, 100, 1), 'n_estimators': hp.quniform('ne', 200, 1000, 50)}
    elif model_name == 'CatBoost':
        return {'learning_rate': hp.loguniform('lr', np.log(0.01), np.log(0.2)), 'depth': hp.quniform('d', 4, 10, 1), 'iterations': hp.quniform('it', 300, 1000, 50)}
    elif model_name == 'SVR':
        return {'C': hp.loguniform('C', np.log(0.1), np.log(100)), 'gamma': hp.choice('gamma', ['scale', 'auto', 0.1, 0.01])}
    elif model_name == 'KNN':
        return {'n_neighbors': hp.quniform('n', 3, 30, 1)}
    elif model_name == 'ExtraTrees':
        return {'n_estimators': hp.quniform('n_est', 100, 1000, 50), 'max_depth': hp.quniform('md', 5, 30, 1), 'min_samples_split': hp.quniform('min_samp', 2, 10, 1)}
    return {}

def optimize_model(model_name, X_train, y_train, X_val, y_val, evals=10):
    space = get_hyperopt_space(model_name)
    def objective(params):
        for p in ['max_depth', 'n_estimators', 'num_leaves', 'depth', 'iterations', 'n_neighbors', 'min_samples_split']:
            if p in params: params[p] = int(params[p])
        
        if model_name == 'XGBoost': model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
        elif model_name == 'LightGBM': model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
        elif model_name == 'CatBoost': model = CatBoostRegressor(**params, random_seed=42, verbose=0, allow_writing_files=False)
        elif model_name == 'KNN': model = KNeighborsRegressor(**params, n_jobs=-1)
        elif model_name == 'SVR': 
            svr = SVR(**params)
            model = TransformedTargetRegressor(regressor=svr, transformer=StandardScaler())
        elif model_name == 'ExtraTrees':
            model = ExtraTreesRegressor(**params, random_state=42, n_jobs=-1)
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        loss = np.sqrt(mean_squared_error(y_val, preds))
        return {'loss': loss, 'status': STATUS_OK, 'model': model}

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=evals, trials=trials)
    best_trial = min(trials.results, key=lambda x: x['loss'])
    return best_trial['model']

# LSTM
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_lstm(X_train, y_train, X_predict, X_dim):
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    X_t = torch.FloatTensor(X_train).unsqueeze(1)
    y_t = torch.FloatTensor(y_scaled)
    X_p_t = torch.FloatTensor(X_predict).unsqueeze(1)
    
    model = LSTMNet(X_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    
    model.train()
    for _ in range(80):
        optimizer.zero_grad()
        loss_fn(model(X_t), y_t).backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        preds = model(X_p_t).numpy()
    return scaler_y.inverse_transform(preds).flatten()

# Statistika (SARIMAX & TBATS)
def train_sarimax(y_train, X_train, X_val, X_test):
    print("  -> Fitting SARIMAX...")
    model = SARIMAX(y_train, exog=X_train, order=(1, 1, 1), seasonal_order=(1, 0, 0, 7), 
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    X_forecast = np.concatenate([X_val, X_test])
    start_idx = len(y_train)
    end_idx = start_idx + len(X_forecast) - 1
    full_preds = results.predict(start=start_idx, end=end_idx, exog=X_forecast)
    return full_preds

def train_tbats(y_train, y_val_test_len):
    """NOVÉ: TBATS Model pro komplexní sezónnost"""
    print("  -> Fitting TBATS (this is slow)...")
    # TBATS si sám řeší sezónnost (weekly, yearly)
    estimator = TBATS(seasonal_periods=[7, 365.25])
    model = estimator.fit(y_train)
    forecast = model.forecast(steps=y_val_test_len)
    return forecast

# Neural Prophet
def run_neural_prophet(full_df, split_date, target_col):
    if not NEURALPROPHET_AVAILABLE: return None
    print("--- Training NeuralProphet ---")
    df_np = full_df[['date', target_col, 'temperature_2m', 'temp_effective', 'is_working_day', 'wind_speed_10m']].copy()
    df_np.rename(columns={'date': 'ds', target_col: 'y'}, inplace=True)
    
    m = NeuralProphet(n_lags=7, n_forecasts=1, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                      batch_size=64, epochs=30, learning_rate=0.01, trainer_config={'accelerator': 'cpu'})
    for reg in ['temperature_2m', 'temp_effective', 'is_working_day', 'wind_speed_10m']: m.add_future_regressor(reg)
        
    train_df_np = df_np[df_np['ds'] < split_date]
    m.fit(train_df_np, freq='D')
    
    future = m.make_future_dataframe(df_np, periods=0, n_historic_predictions=True)
    forecast = m.predict(future)
    forecast = forecast[['ds', 'yhat1']].rename(columns={'ds': 'date', 'yhat1': 'pred'})
    
    merged = pd.merge(full_df[['date']], forecast, on='date', how='left')
    result_slice = merged[merged['date'] >= split_date]['pred'].values
    
    if np.isnan(result_slice).any():
        mask = np.isnan(result_slice)
        result_slice[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), result_slice[~mask])
    return result_slice

# Deep Learning (TFT & N-HiTS)
if TFT_AVAILABLE:
    class PLWrapper(pl.LightningModule):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.model.log = self.safe_log 
        def safe_log(self, name, value, **kwargs):
            if value is None: return
            self.log(name, value, **kwargs)
        def forward(self, x): return self.model(x)
        def training_step(self, b, i): 
            self.model.trainer = self.trainer; return self.model.training_step(b, i)
        def validation_step(self, b, i): 
            self.model.trainer = self.trainer; return self.model.validation_step(b, i)
        def configure_optimizers(self): return self.model.configure_optimizers()

def run_deep_model(full_df, split_date, target_col, model_type='TFT', max_epochs=30):
    if not TFT_AVAILABLE: return np.zeros(len(full_df[full_df['date'] >= split_date]))
    print(f"--- Training {model_type} (Integrated Fix V3) ---")
    data = full_df.copy()
    
    data['target_log'] = np.log1p(data[target_col])
    train_indices = data[data['date'] < split_date].index
    scaler = StandardScaler()
    scaler.fit(data.loc[train_indices, 'target_log'].values.reshape(-1, 1))
    data['target_scaled'] = scaler.transform(data[['target_log']].values).flatten()
    
    for c in ["is_weekend", "is_working_day", "is_holiday"]: data[c] = data[c].astype(str)
    data['time_idx'] = (data['date'] - data['date'].min()).dt.days
    data['group_id'] = "CZ"
    data['month'] = data['date'].dt.month.astype(str)
    
    cutoff_idx = data[data['date'] >= split_date].iloc[0]['time_idx']
    training_cutoff = cutoff_idx - 1
    
    max_encoder_length = 60
    if model_type == 'N-HiTS':
        min_encoder_length = 60 
        add_relative_time_idx = False; add_target_scales = False
    else:
        min_encoder_length = 30 
        add_relative_time_idx = True; add_target_scales = True

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx", target="target_scaled", group_ids=["group_id"],
        min_encoder_length=min_encoder_length, max_encoder_length=max_encoder_length,
        min_prediction_length=1, max_prediction_length=1,
        static_categoricals=["group_id"],
        time_varying_known_categoricals=["is_weekend", "is_working_day", "is_holiday", "month"],
        time_varying_known_reals=["time_idx", "day_of_year", "temperature_2m", "temp_effective"],
        time_varying_unknown_reals=["target_scaled"],
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation=None), 
        add_relative_time_idx=add_relative_time_idx, add_target_scales=add_target_scales, add_encoder_length=True,
    )
    
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)

    if model_type == 'TFT':
        model_base = TemporalFusionTransformer.from_dataset(training, learning_rate=0.005, hidden_size=64, attention_head_size=4, dropout=0.1, hidden_continuous_size=16, output_size=7, loss=QuantileLoss())
    elif model_type == 'N-HiTS':
        model_base = NHiTS.from_dataset(training, learning_rate=0.001, weight_decay=1e-2, loss=QuantileLoss(), backcast_loss_ratio=0.0)
    
    model = PLWrapper(model_base)
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu", gradient_clip_val=0.1, enable_checkpointing=True, logger=False)
    trainer.barebones = False
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    best_path = trainer.checkpoint_callback.best_model_path
    print(f"Loading best {model_type} model from: {best_path}")
    
    checkpoint = torch.load(best_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."): new_state_dict[k[6:]] = v 
        else: new_state_dict[k] = v
    model_base.load_state_dict(new_state_dict)
    
    best_model = PLWrapper(model_base)
    best_model.model.trainer = trainer; best_model.trainer = trainer
    
    test_start_row = data[data['date'] >= split_date].index[0]
    inference_start_idx = max(0, test_start_row - max_encoder_length)
    inference_data = data.iloc[inference_start_idx:].reset_index(drop=True)
    inference_dataset = TimeSeriesDataSet.from_dataset(training, inference_data, predict=False, stop_randomization=True)
    inference_dataloader = inference_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
    
    raw_predictions = best_model.model.predict(inference_dataloader, mode="prediction", return_x=True)
    preds_scaled = raw_predictions.output.cpu().numpy().flatten()
    time_indices = raw_predictions.x["decoder_time_idx"].cpu().numpy().flatten()
    
    preds_log = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    preds_final = np.expm1(preds_log)
    
    res_df = pd.DataFrame({'time_idx': time_indices.astype(int), 'pred': preds_final})
    pred_map = dict(zip(res_df.time_idx, res_df.pred))
    final_preds = []
    target_indices = data[data['date'] >= split_date]['time_idx'].values.astype(int)
    for tidx in target_indices:
        val = pred_map.get(tidx, np.nan)
        final_preds.append(val)
        
    print(f"{model_type} Alignment: Matched {len([x for x in final_preds if not np.isnan(x)])}/{len(target_indices)} days.")
    preds_array = np.array(final_preds)
    if np.isnan(preds_array).any():
        mask = np.isnan(preds_array)
        preds_array[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), preds_array[~mask])
    return preds_array

# ==========================================
# 4. ENSEMBLE LOGIC
# ==========================================
class ContextualBanditEnsemble:
    def __init__(self, n_arms, n_features, alpha=0.1):
        self.n_arms = n_arms; self.alpha = alpha
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]
    def select_arm(self, context_vector):
        p = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            try: theta = np.linalg.solve(self.A[a], self.b[a])
            except: theta = np.linalg.inv(self.A[a]) @ self.b[a]
            mean = theta @ context_vector
            var = context_vector.T @ np.linalg.inv(self.A[a]) @ context_vector
            p[a] = mean + self.alpha * np.sqrt(var)
        return np.argmax(p)
    def update_all(self, context_vector, rewards):
        for a in range(self.n_arms):
            self.A[a] += np.outer(context_vector, context_vector); self.b[a] += rewards[a] * context_vector

def run_contextual_bandits_online(full_preds_df, test_start_date, model_cols, target_col='Actual', features_cols=None):
    print("--- Running Online Contextual Bandits ---")
    df = full_preds_df.sort_values('date').reset_index(drop=True)
    test_indices = df[df['date'] >= test_start_date].index
    if features_cols is None:
        possible_feats = ['temperature_2m', 'temp_effective', 'day_of_year', 'is_weekend', 'wind_speed_10m']
        features_cols = [c for c in possible_feats if c in df.columns]
    n_models = len(model_cols); n_features = len(features_cols) + 1
    bandit = ContextualBanditEnsemble(n_models, n_features, alpha=0.5)
    warmup_start = max(0, test_indices[0] - 60)
    for idx in range(warmup_start, test_indices[0]):
        ctx = np.append(df.loc[idx, features_cols].values.astype(float), 1.0)
        rewards = -np.abs(df.loc[idx, model_cols].values - df.loc[idx, target_col])
        bandit.update_all(ctx, rewards)
    ensemble_preds = []
    for idx in test_indices:
        ctx = np.append(df.loc[idx, features_cols].values.astype(float), 1.0)
        chosen = bandit.select_arm(ctx)
        ensemble_preds.append(df.loc[idx, model_cols[chosen]])
        rewards = -np.abs(df.loc[idx, model_cols].values - df.loc[idx, target_col])
        bandit.update_all(ctx, rewards)
    return np.array(ensemble_preds)

def run_contextual_bandits_rolling(full_preds_df, test_start_date, model_cols, target_col='Actual', features_cols=None, strategy='month'):
    print(f"--- Running Rolling Bandits ({strategy}) ---")
    df = full_preds_df.sort_values('date').reset_index(drop=True)
    test_indices = df[df['date'] >= test_start_date].index
    if features_cols is None:
        possible_feats = ['temperature_2m', 'temp_effective', 'day_of_year', 'is_weekend', 'wind_speed_10m']
        features_cols = [c for c in possible_feats if c in df.columns]
    n_models = len(model_cols); n_features = len(features_cols) + 1
    ensemble_preds = []
    for idx in test_indices:
        target_date = df.loc[idx, 'date']
        data_avail_until = target_date - timedelta(days=2)
        if strategy == 'week': win_start = data_avail_until - timedelta(days=6)
        else: win_start = data_avail_until - timedelta(days=30)
        mask = (df['date'] >= win_start) & (df['date'] <= data_avail_until)
        hist_df = df.loc[mask].copy()
        if strategy == 'weighted_month':
            w_start = data_avail_until - timedelta(days=6)
            w_df = hist_df.loc[hist_df['date'] >= w_start]
            hist_df = pd.concat([hist_df, w_df, w_df], axis=0)
        if len(hist_df) < 5: hist_df = df.loc[df['date'] <= data_avail_until]
        bandit = ContextualBanditEnsemble(n_models, n_features, alpha=0.2)
        X_ctx = np.c_[hist_df[features_cols].values.astype(float), np.ones(len(hist_df))]
        y_act = hist_df[target_col].values
        m_preds = hist_df[model_cols].values
        rewards_matrix = -np.abs(m_preds - y_act[:, None])
        for i in range(len(hist_df)):
            bandit.update_all(X_ctx[i], rewards_matrix[i])
        curr_ctx = np.append(df.loc[idx, features_cols].values.astype(float), 1.0)
        chosen = bandit.select_arm(curr_ctx)
        ensemble_preds.append(df.loc[idx, model_cols[chosen]])
    return np.array(ensemble_preds)

def solve_weights(X, y, method='frank_wolfe'):
    n_samples, n_models = X.shape
    if method == 'frank_wolfe':
        w = np.ones(n_models) / n_models
        for k in range(500):
            resid = np.dot(X, w) - y; grad = (2/n_samples) * np.dot(X.T, resid)
            s = np.zeros(n_models); s[np.argmin(grad)] = 1.0
            gamma = 2.0 / (k + 2.0); w = (1 - gamma) * w + gamma * s
        return w
    elif method == 'nnls':
        w, _ = nnls(X, y); 
        return w / np.sum(w) if np.sum(w) > 0 else np.ones(n_models) / n_models
    elif method == 'ensemble_selection':
        pool = []
        best_single = np.argmin([mean_squared_error(y, X[:, i]) for i in range(n_models)])
        pool.append(best_single); curr_pred = X[:, best_single]
        for _ in range(20): 
            best_idx, best_err = -1, float('inf')
            for i in range(n_models):
                tmp = (curr_pred * len(pool) + X[:, i]) / (len(pool) + 1)
                err = mean_squared_error(y, tmp)
                if err < best_err: best_err = err; best_idx = i
            pool.append(best_idx); curr_pred = (curr_pred * (len(pool)-1) + X[:, best_idx]) / len(pool)
        w = np.zeros(n_models); 
        for idx in pool: w[idx] += 1
        return w / np.sum(w)
    return np.ones(n_models) / n_models

def get_rolling_weights(full_preds_df, test_start_date, model_cols, target_col='Actual', strategy='month', method='frank_wolfe'):
    df = full_preds_df.sort_values('date').reset_index(drop=True)
    test_indices = df[df['date'] >= test_start_date].index
    preds = []
    for idx in test_indices:
        target_date = df.loc[idx, 'date']
        data_avail_until = target_date - timedelta(days=2) 
        if strategy == 'week': win_start = data_avail_until - timedelta(days=6) 
        else: win_start = data_avail_until - timedelta(days=30)
        mask = (df['date'] >= win_start) & (df['date'] <= data_avail_until)
        hist_df = df.loc[mask].copy()
        if strategy == 'weighted_month':
            w_start = data_avail_until - timedelta(days=6)
            w_df = hist_df.loc[hist_df['date'] >= w_start]
            hist_df = pd.concat([hist_df, w_df, w_df], axis=0)
        if len(hist_df) < 5: hist_df = df.loc[df['date'] <= data_avail_until]
        X_h = hist_df[model_cols].values; y_h = hist_df[target_col].values
        w_opt = solve_weights(X_h, y_h, method=method)
        preds.append(np.dot(df.loc[idx, model_cols].values, w_opt))
    return np.array(preds)

def plot_weights_evolution(full_preds_df, test_start_date, model_cols, target_col='Actual', folder='output_graphs', method='frank_wolfe', strategy='month'):
    df = full_preds_df.sort_values('date').reset_index(drop=True)
    test_indices = df[df['date'] >= test_start_date].index
    weights_history = []; dates = []
    for idx in test_indices:
        target_date = df.loc[idx, 'date']
        data_avail_until = target_date - timedelta(days=2)
        if strategy == 'week': win_start = data_avail_until - timedelta(days=6)
        else: win_start = data_avail_until - timedelta(days=30)
        mask = (df['date'] >= win_start) & (df['date'] <= data_avail_until)
        hist_df = df.loc[mask].copy()
        if strategy == 'weighted_month':
            w_start = data_avail_until - timedelta(days=6)
            w_df = hist_df.loc[hist_df['date'] >= w_start]
            hist_df = pd.concat([hist_df, w_df, w_df], axis=0)
        if len(hist_df) < 5: hist_df = df.loc[df['date'] <= data_avail_until]
        X_h = hist_df[model_cols].values; y_h = hist_df[target_col].values
        w_opt = solve_weights(X_h, y_h, method=method)
        weights_history.append(w_opt); dates.append(target_date)
    weights_df = pd.DataFrame(weights_history, columns=model_cols, index=dates)
    plt.figure(figsize=(18, 9))
    plt.stackplot(weights_df.index, weights_df.T, labels=weights_df.columns, alpha=0.85)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(f'Weights Evolution ({method} | {strategy})', fontsize=16)
    plt.ylabel('Weight', fontsize=12); plt.xlabel('Date', fontsize=12)
    plt.margins(0, 0); plt.tight_layout()
    plt.savefig(os.path.join(folder, f"weights_evolution_{method}_{strategy}.png"), dpi=300); plt.close()
    weights_df.to_csv(os.path.join(CONFIG['output_data'], f"weights_history_{method}_{strategy}.csv"))

# ==========================================
# 5. REPORTING (NOVÉ: SUBPLOTS)
# ==========================================

def generate_report(test_df, cols, title, folder, filename):
    metrics = []
    y_t = test_df['Actual']
    for m in cols:
        if m not in test_df.columns: continue
        y_p = test_df[m]
        metrics.append({
            'Model': m, 
            'MAE': mean_absolute_error(y_t, y_p), 
            'RMSE': np.sqrt(mean_squared_error(y_t, y_p)), 
            'MAPE (%)': np.mean(np.abs((y_t - y_p) / y_t)) * 100
        })
    df_m = pd.DataFrame(metrics).sort_values('RMSE')
    print(f"\n--- {title} ---")
    print(df_m)
    df_m.to_latex(os.path.join(folder, f"{filename}.tex"), index=False, float_format="%.2f")
    
    plt.figure(figsize=(14, 7))
    plt.plot(test_df['date'], test_df['Actual'], 'k-', lw=2, label='Actual', alpha=0.8)
    cmap = plt.get_cmap('tab10')
    for i, c in enumerate(cols):
        if c in test_df.columns:
            plt.plot(test_df['date'], test_df[c], ls='--', alpha=0.7, label=c, color=cmap(i))
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{filename}.png"), dpi=300)
    plt.close()

def generate_grouped_report(test_df, model_cols, folder):
    # Definice skupin
    groups = {
        "Tree Models": ['XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees'],
        "Deep Learning": ['TFT', 'N-HiTS', 'NeuralProphet', 'LSTM'],
        "Statistical": ['Prophet', 'SARIMAX', 'TBATS'],
        "Classical ML": ['SVR', 'ElasticNet', 'KNN', 'BayesianRidge']
    }
    
    # 1. Tabulka metrik
    metrics = []
    y_true = test_df['Actual']
    for m in model_cols:
        if m not in test_df.columns: continue
        y_p = test_df[m]
        metrics.append({
            'Model': m, 
            'MAE': mean_absolute_error(y_true, y_p), 
            'RMSE': np.sqrt(mean_squared_error(y_true, y_p)), 
            'MAPE': np.mean(np.abs((y_true - y_p) / y_true)) * 100
        })
    df_m = pd.DataFrame(metrics).sort_values('MAPE')
    print("\n--- Final Model Leaderboard ---")
    print(df_m)
    df_m.to_csv(os.path.join(folder, "final_leaderboard.csv"), index=False)
    
    # 2. Subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharex=True)
    axes = axes.flatten()
    
    for i, (grp_name, models) in enumerate(groups.items()):
        ax = axes[i]
        ax.plot(test_df['date'], test_df['Actual'], 'k-', lw=1.5, label='Actual', alpha=0.6)
        
        # Color cycle
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for j, m in enumerate(models):
            if m in test_df.columns:
                # Calculate MAPE for label
                mape = np.mean(np.abs((y_true - test_df[m]) / y_true)) * 100
                ax.plot(test_df['date'], test_df[m], ls='--', lw=1.5, label=f"{m} ({mape:.1f}%)", color=colors[j])
        
        ax.set_title(grp_name, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        if i >= 2: ax.set_xlabel("Date")
        if i % 2 == 0: ax.set_ylabel("Gas Consumption")
        
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "report_base_models_grouped.png"), dpi=300)
    plt.close()

def main():
    df = prepare_dataset()
    test_start = pd.to_datetime(CONFIG['split_test_date'])
    val_start = pd.to_datetime(CONFIG['split_val_date'])
    train_df = df[df['date'] < val_start]
    val_df = df[(df['date'] >= val_start) & (df['date'] < test_start)]
    test_df = df[df['date'] >= test_start]
    
    feats = [c for c in df.columns if c not in ['date', CONFIG['target_col']]]
    target = CONFIG['target_col']
    X_train = train_df[feats].values; y_train = train_df[target].values
    X_val = val_df[feats].values; y_val = val_df[target].values
    X_test = test_df[feats].values
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train); X_val_sc = scaler.transform(X_val); X_test_sc = scaler.transform(X_test)
    
    full_preds = pd.concat([val_df, test_df])[['date', target]].copy()
    full_preds.rename(columns={target: 'Actual'}, inplace=True); full_preds.reset_index(drop=True, inplace=True)
    def predict_all(model, is_scaled=False):
        p_val = model.predict(X_val_sc) if is_scaled else model.predict(X_val)
        p_test = model.predict(X_test_sc) if is_scaled else model.predict(X_test)
        return np.concatenate([p_val, p_test])

    print("\n--- Training Models ---")
    # Tree
    for m in ['XGBoost', 'LightGBM', 'ExtraTrees']:
        mod = optimize_model(m, X_train, y_train, X_val, y_val, CONFIG['hyperopt_evals'])
        full_preds[m] = predict_all(mod)
    if CATBOOST_AVAILABLE:
        mod = optimize_model('CatBoost', X_train, y_train, X_val, y_val, CONFIG['hyperopt_evals'])
        full_preds['CatBoost'] = predict_all(mod)
        
    # Classical
    for m in ['KNN', 'SVR']:
        mod = optimize_model(m, X_train_sc, y_train, X_val_sc, y_val, CONFIG['hyperopt_evals'])
        full_preds[m] = predict_all(mod, is_scaled=True)
    enet = ElasticNet(alpha=0.1); enet.fit(X_train_sc, y_train); full_preds['ElasticNet'] = predict_all(enet, is_scaled=True)
    bay = BayesianRidge(); bay.fit(X_train_sc, y_train); full_preds['BayesianRidge'] = predict_all(bay, is_scaled=True) # NOVÉ
    
    # DL & Stats
    full_preds['LSTM'] = np.concatenate([train_lstm(X_train_sc, y_train, X_val_sc, X_train.shape[1]), train_lstm(X_train_sc, y_train, X_test_sc, X_train.shape[1])])
    if PROPHET_AVAILABLE:
        p_df = train_df[['date', target, 'is_working_day', 'temperature_2m']].rename(columns={'date':'ds', target:'y'})
        m = Prophet(); m.add_regressor('is_working_day'); m.add_regressor('temperature_2m'); m.fit(p_df)
        fut = pd.concat([val_df, test_df])[['date', 'is_working_day', 'temperature_2m']].rename(columns={'date':'ds'})
        full_preds['Prophet'] = m.predict(fut)['yhat'].values
    if NEURALPROPHET_AVAILABLE:
        np_p = run_neural_prophet(df, val_start, target)
        if np_p is not None and len(np_p) == len(full_preds): full_preds['NeuralProphet'] = np_p
    if SARIMAX_AVAILABLE: full_preds['SARIMAX'] = train_sarimax(y_train, X_train, X_val, X_test)
    if TBATS_AVAILABLE: 
        # TBATS needs only Y history and forecast length
        print("  -> Fitting TBATS...")
        # Fit on train, forecast Val+Test
        tb_mod = TBATS(seasonal_periods=[7, 365.25]).fit(y_train)
        full_preds['TBATS'] = tb_mod.forecast(steps=len(full_preds))

    if TFT_AVAILABLE:
        tft = run_deep_model(df, val_start, target, 'TFT', CONFIG['tft_epochs'])
        if len(tft) == len(full_preds): full_preds['TFT'] = tft
        nhits = run_deep_model(df, val_start, target, 'N-HiTS', CONFIG['tft_epochs'])
        if len(nhits) == len(full_preds): full_preds['N-HiTS'] = nhits

    # Ensemble
    print("\n--- Generating Ensembles ---")
    model_cols = [c for c in full_preds.columns if c not in ['date', 'Actual']]
    test_res = full_preds[full_preds['date'] >= test_start].copy()
    test_res['Ensemble_Avg'] = test_res[model_cols].mean(axis=1)
    
    methods = ['frank_wolfe', 'nnls', 'ensemble_selection']
    windows = [('week', 'a_Last_Week'), ('month', 'b_Last_Month'), ('weighted_month', 'c_Weighted_Month')]
    for method in methods:
        for strat, label in windows:
            test_res[f"{method}_{label}"] = get_rolling_weights(full_preds, test_start, model_cols, strategy=strat, method=method)

    test_res['Bandit_Online'] = run_contextual_bandits_online(full_preds, test_start, model_cols, target_col='Actual')
    for strat in ['week', 'month', 'weighted_month']:
        test_res[f"Bandit_Rolling_{strat}"] = run_contextual_bandits_rolling(full_preds, test_start, model_cols, target_col='Actual', strategy=strat)

    # Reporting
    out_dir = CONFIG['output_graphs']
    # 1. Nový Grouped Report (Subplots)
    generate_grouped_report(test_res, model_cols, out_dir)
    
    # 2. Ensemble Reports
    for label, title in [('Last_Week', 'Last Week Data'), ('Last_Month', 'Last Month Data'), ('Weighted_Month', 'Weighted Month')]:
        cols = [c for c in test_res.columns if label in c] + ['Ensemble_Avg']
        generate_report(test_res, cols, f"Ensemble Opt: {title}", out_dir, f"report_2_{label.lower()}")
        
    bandit_cols = [c for c in test_res.columns if 'Bandit' in c] + ['Ensemble_Avg']
    generate_report(test_res, bandit_cols, "Contextual Bandits Comparison", out_dir, "report_3_bandits")

    for method in methods:
        for strat in ['week', 'month', 'weighted_month']:
            plot_weights_evolution(full_preds, test_start, model_cols, target_col='Actual', folder=out_dir, method=method, strategy=strat)

    test_res.to_csv(os.path.join(CONFIG['output_data'], 'final_results_complete.csv'), index=False)
    print(f"\nAll Done. Results in {out_dir}")

if __name__ == "__main__":
    start_time = time.time()
    print(f"\n--- Script execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    main()

    end_time = time.time()
    duration = end_time - start_time
    print(f"--- Script execution finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"Total execution time: {duration/60:.2f} minutes ({duration:.2f} seconds)")