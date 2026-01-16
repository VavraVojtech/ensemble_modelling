import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import warnings
import holidays

# Vypnutí warningů
warnings.filterwarnings('ignore')
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# ==========================================
# 1. KONFIGURACE
# ==========================================
CONFIG = {
    'input_folder': 'input_data',
    'target_col': 'spotreba_cr',
    'split_val_date': '2024-10-01',   
    'split_test_date': '2024-12-01',  
    'seed': 42,
    'tft_epochs': 30,
    'batch_size': 64,
    'learning_rate': 0.005, 
    'hidden_size': 64,
    'attn_heads': 4
}

pl.seed_everything(CONFIG['seed'])

# ==========================================
# 2. DEFINICE WRAPPERU (FIXED)
# ==========================================
class TFTWrapper(pl.LightningModule):
    def __init__(self, tft_model):
        super().__init__()
        self.model = tft_model
        # Fix logování
        self.model.log = self.safe_log

    def safe_log(self, name, value, **kwargs):
        if value is None: return
        self.log(name, value, **kwargs)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # --- FIX: Předání trenéra vnitřnímu modelu ---
        self.model.trainer = self.trainer
        return self.model.training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        # --- FIX: Předání trenéra vnitřnímu modelu ---
        self.model.trainer = self.trainer
        return self.model.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        return self.model.configure_optimizers()

# ==========================================
# 3. PŘÍPRAVA DAT
# ==========================================
def prepare_dataset():
    print("--- Loading Data ---")
    path_ote = os.path.join(CONFIG['input_folder'], 'OTE_NG_ODCHYLKY_NC_BAL.csv')
    path_weather = os.path.join(CONFIG['input_folder'], 'OpenMeteo_train_data_6_6.csv')
    
    if not os.path.exists(path_ote) or not os.path.exists(path_weather):
        raise FileNotFoundError(f"Input files missing in {CONFIG['input_folder']}")

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

    # Feature Engineering
    temps = df['temperature_2m'].values
    t_eff = np.zeros_like(temps)
    t_eff[0] = temps[0]
    for t in range(1, len(temps)):
        t_eff[t] = 0.5 * temps[t] + 0.5 * t_eff[t-1]
    df['temp_effective'] = t_eff
    
    cz_holidays = holidays.CZ()
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(str)
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in cz_holidays else 0).astype(str)
    
    df['day_of_year'] = df['date'].dt.dayofyear.astype(float)
    df['month'] = df['date'].dt.month.astype(str)
    df['time_idx'] = (df['date'] - df['date'].min()).dt.days
    df['group_id'] = "CZ" 
    
    return df

# ==========================================
# 4. HLAVNÍ LOOP
# ==========================================
def main():
    df = prepare_dataset()
    
    train_end_idx = df[df['date'] < pd.to_datetime(CONFIG['split_val_date'])].index[-1]
    
    # --- MANUÁLNÍ LOG TRANSFORMACE ---
    df['target_log'] = np.log1p(df[CONFIG['target_col']])
    
    # Standardizace
    scaler = StandardScaler()
    train_subset = df.loc[:train_end_idx, 'target_log'].values.reshape(-1, 1)
    scaler.fit(train_subset)
    df['target_scaled'] = scaler.transform(df[['target_log']].values).flatten()
    
    print(f"Log-Target Mean (Train): {scaler.mean_[0]:.4f}, Std: {np.sqrt(scaler.var_[0]):.4f}")

    training_cutoff = df.loc[train_end_idx, "time_idx"]
    max_encoder_length = 60 

    # --- DEFINICE DATASETU ---
    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="target_scaled", 
        group_ids=["group_id"],
        min_encoder_length=30,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=["group_id"],
        time_varying_known_categoricals=["is_weekend", "is_holiday", "month"],
        time_varying_known_reals=["time_idx", "day_of_year", "temperature_2m", "temp_effective"],
        time_varying_unknown_reals=["target_scaled"],
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation=None),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    train_dataloader = training.to_dataloader(train=True, batch_size=CONFIG['batch_size'], num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=CONFIG['batch_size']*2, num_workers=0)

    # --- MODEL TFT ---
    tft_base = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=CONFIG['learning_rate'],
        hidden_size=CONFIG['hidden_size'],
        attention_head_size=CONFIG['attn_heads'],
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7, 
        loss=QuantileLoss(),
    )
    
    # ZABALENÍ DO WRAPPERU
    tft = TFTWrapper(tft_base)

    trainer = pl.Trainer(
        max_epochs=CONFIG['tft_epochs'],
        accelerator="cpu",
        gradient_clip_val=0.1,
        enable_checkpointing=True,
        logger=False
    )
    
    trainer.barebones = False

    print("--- Starting Training ---")
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # --- PREDIKCE ---
    print("--- Predicting ---")
    
    tft.model.trainer = trainer
    
    raw_predictions = tft.model.predict(val_dataloader, mode="prediction", return_x=True)
    preds_scaled = raw_predictions.output.cpu().numpy().flatten()
    
    # Inverzní transformace
    preds_log = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    preds_original = np.expm1(preds_log)
    
    # Skutečné hodnoty
    actuals_scaled = raw_predictions.x["decoder_target"].cpu().numpy().flatten()
    actuals_log = scaler.inverse_transform(actuals_scaled.reshape(-1, 1)).flatten()
    actuals_original = np.expm1(actuals_log)
    
    # Metriky
    rmse = np.sqrt(mean_squared_error(actuals_original, preds_original))
    mae = mean_absolute_error(actuals_original, preds_original)
    mape = np.mean(np.abs((actuals_original - preds_original) / actuals_original)) * 100
    
    print(f"\n--- TFT RESULTS (Validation/Test) ---")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE:  {mae:,.2f}")
    print(f"MAPE: {mape:.2f} %")

    # Plot
    plt.figure(figsize=(15, 6))
    subset = 300 
    plt.plot(actuals_original[-subset:], label="Actual", color="black", linewidth=2)
    plt.plot(preds_original[-subset:], label="Prediction", color="red", linestyle="--", linewidth=2)
    plt.title(f"TFT Check - MAPE: {mape:.2f}%")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("check_tft_prediction.png")
    print("Graph saved to check_tft_prediction.png")

if __name__ == "__main__":
    main()