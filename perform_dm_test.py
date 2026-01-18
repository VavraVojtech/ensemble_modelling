import pandas as pd
import numpy as np
from scipy import stats
import os

# ==========================================
# KONFIGURACE
# ==========================================
INPUT_FILE = 'output_data/final_results_complete.csv'
TARGET_COL = 'Actual'

# Zde definujte názvy sloupců dvou modelů, které chcete porovnat
# (Podle vašich výsledků jsou to vítězné strategie pro 6 týdnů)
MODEL_1 = 'frank_wolfe_6_weeks_weighted'
MODEL_2 = 'ensemble_selection_6_weeks_weighted'

def diebold_mariano_test(y_true, y_pred1, y_pred2, h=1, criterion="MAE"):
    """
    Vypočítá Diebold-Mariano test pro porovnání přesnosti predikcí.
    
    H0: Oba modely mají stejnou přesnost.
    H1: Modely mají rozdílnou přesnost.
    
    Args:
        y_true: Skutečné hodnoty
        y_pred1: Predikce modelu 1
        y_pred2: Predikce modelu 2
        h: Horizont predikce (pro denní data obvykle 1)
        criterion: "MAE" (Mean Absolute Error) nebo "MSE" (Mean Squared Error)
    
    Returns:
        dm_stat: DM statistika
        p_value: p-hodnota
    """
    # 1. Výpočet chyb (residuals)
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    
    # 2. Výpočet diferenciálu ztrátové funkce (Loss Differential)
    if criterion == "MSE":
        d = (e1**2) - (e2**2)
    elif criterion == "MAE":
        d = np.abs(e1) - np.abs(e2)
    elif criterion == "MAPE":
        d = (np.abs(e1) / y_true) - (np.abs(e2) / y_true)
    
    # 3. Průměr diferenciálu
    mean_d = np.mean(d)
    
    # 4. Výpočet autokovariance (pro robustní rozptyl)
    T = float(len(d))
    autocov = np.var(d, ddof=0) # gamma_0
    
    for lag in range(1, h):
        gamma = np.cov(d[:-lag], d[lag:])[0, 1]
        autocov += 2 * gamma
        
    # 5. DM Statistika
    # Vzorec: mean_d / sqrt(variance / T)
    dm_stat = mean_d / np.sqrt(autocov / T)
    
    # 6. p-hodnota (oboustranný test, normální rozdělení)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"CHYBA: Soubor '{INPUT_FILE}' nebyl nalezen.")
        return

    print(f"--- Načítám data: {INPUT_FILE} ---")
    df = pd.read_csv(INPUT_FILE)
    
    # Kontrola existence sloupců
    if MODEL_1 not in df.columns or MODEL_2 not in df.columns:
        print("CHYBA: Jeden z modelů nebyl v CSV nalezen. Zkontrolujte názvy sloupců.")
        print("Dostupné sloupce:", df.columns.tolist())
        return

    # Odstranění NaN (pokud existují např. na začátku)
    df_clean = df[[TARGET_COL, MODEL_1, MODEL_2]].dropna()
    
    y_true = df_clean[TARGET_COL].values
    y_p1 = df_clean[MODEL_1].values
    y_p2 = df_clean[MODEL_2].values

    print(f"Srovnávám:\n 1) {MODEL_1}\n 2) {MODEL_2}")
    print(f"Počet pozorování: {len(y_true)}")
    
    # Spuštění testu (používáme MAE/MAPE logiku, tj. absolutní chybu)
    dm_stat, p_value = diebold_mariano_test(y_true, y_p1, y_p2, h=1, criterion="MAE")
    
    print("\n--- Výsledky Diebold-Mariano Testu ---")
    print(f"DM Statistika: {dm_stat:.4f}")
    print(f"p-hodnota:     {p_value:.4f}")
    
    print("\n--- Interpretace ---")
    if p_value < 0.05:
        print(">> Rozdíl JE statisticky významný (zamítáme H0).")
        if dm_stat < 0:
            print(f">> Model 1 ({MODEL_1}) je signifikantně LEPŠÍ.")
        else:
            print(f">> Model 2 ({MODEL_2}) je signifikantně LEPŠÍ.")
    else:
        print(">> Rozdíl NENÍ statisticky významný (nezamítáme H0).")
        print(">> Oba modely mají srovnatelnou přesnost.")
        print(">> To podporuje argument pro výběr stabilnějšího modelu (Frank-Wolfe).")

if __name__ == "__main__":
    main()