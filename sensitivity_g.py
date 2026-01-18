import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# ==========================================
# KONFIGURACE
# ==========================================
INPUT_FILE = 'output_data/final_results_complete.csv'
OUTPUT_DIR = 'output_graph'
OUTPUT_FILE = 'sensitivity_analysis.png'
TARGET_COL = 'Actual'

# Nastavení stylu grafu
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 7)})

def calculate_mape(y_true, y_pred):
    """Vypočítá Mean Absolute Percentage Error v procentech."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def main():
    # 1. Načtení dat
    if not os.path.exists(INPUT_FILE):
        print(f"CHYBA: Soubor '{INPUT_FILE}' nebyl nalezen. Spusťte nejdříve ensemble_modelling.py.")
        return

    print(f"Načítám data z {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Kontejner pro výsledky
    results = []

    # 2. Regulární výraz pro identifikaci sloupců Frank-Wolfe
    pattern = r"frank_wolfe_(\d+)_weeks(_weighted|_exponential)?"

    print("Zpracovávám výsledky pro Frank-Wolfe strategie...")
    for col in df.columns:
        match = re.match(pattern, col)
        if match:
            weeks = int(match.group(1))
            suffix = match.group(2)
            
            if suffix == '_weighted':
                variant = 'Recency-Weighted (Heuristic)'
                marker = 'o'
            elif suffix == '_exponential':
                variant = 'Exponential Decay'
                marker = 's'
            else:
                variant = 'Standard (Rolling Window)'
                marker = '^'

            mape = calculate_mape(df[TARGET_COL], df[col])
            
            results.append({
                'Window Length (Weeks)': weeks,
                'Variant': variant,
                'MAPE (%)': mape,
                'Marker': marker
            })

    if not results:
        print("Nebyly nalezeny žádné sloupce odpovídající strategii Frank-Wolfe.")
        return

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by=['Variant', 'Window Length (Weeks)'])

    # 3. Vykreslení grafu
    plt.figure(figsize=(12, 7))
    
    palette = {
        'Standard (Rolling Window)': '#95a5a6',       # Šedá
        'Exponential Decay': '#3498db',               # Modrá
        'Recency-Weighted (Heuristic)': '#e74c3c'     # Červená (hlavní)
    }

    # Lineplot
    sns.lineplot(
        data=res_df, 
        x='Window Length (Weeks)', 
        y='MAPE (%)', 
        hue='Variant', 
        style='Variant', 
        markers=True, 
        dashes=False, 
        palette=palette,
        linewidth=2.5,
        markersize=9
    )

    # --- ANOTACE PRO OBĚ MINIMA ---
    
    # Najdeme data pro Recency-Weighted variantu
    weighted_df = res_df[res_df['Variant'] == 'Recency-Weighted (Heuristic)']
    
    # 1. Lokální minimum (6 weeks)
    row_6 = weighted_df[weighted_df['Window Length (Weeks)'] == 6].iloc[0]
    val_6 = row_6['MAPE (%)']

    # 2. Globální minimum / Sweet Spot (10 weeks)
    row_10 = weighted_df[weighted_df['Window Length (Weeks)'] == 10].iloc[0]
    val_10 = row_10['MAPE (%)']

    # Zvýraznění oblasti stability (6-10 týdnů)
    plt.axvspan(6, 10, color='yellow', alpha=0.1, label='Stable Region (6-10 weeks)')
    
    plt.annotate(
        f"Local Min\n(6 weeks, {val_6:.2f}%)",
        xy=(6, val_6+0.005),
        xytext=(6, val_6 + 0.10),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        fontsize=10, 
        fontweight='bold',
        horizontalalignment='center',
        color='#555555'
    )

    plt.annotate(
        f"Global Min (Sweet Spot)\n(10 weeks, {val_10:.2f}%)",
        xy=(10, val_10),
        xytext=(10, val_10 + 0.15),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        fontsize=11, 
        fontweight='bold',
        horizontalalignment='center'
    )

    # Úpravy grafu
    plt.title('Sensitivity Analysis: Effect of Memory Length on Forecast Accuracy', fontsize=16, pad=15)
    plt.ylabel('MAPE (%)', fontsize=14)
    plt.xlabel('Rolling Window Length (Weeks)', fontsize=14)
    plt.xticks(res_df['Window Length (Weeks)'].unique())
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Legenda (včetně Stable Region)
    handles, labels = plt.gca().get_legend_handles_labels()
    # Přeuspořádání legendy, aby Stable Region byl dole
    order = [0, 1, 2, 3] 
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
               title='Weighting Scheme', fontsize=11, title_fontsize=12, loc='upper right')

    # 5. Uložení
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nGraf úspěšně uložen do: {save_path}")
    
    # Výpis tabulky pro kontrolu
    print("\n--- Data pro graf ---")
    pivot_table = res_df.pivot(index='Window Length (Weeks)', columns='Variant', values='MAPE (%)')
    print(pivot_table.round(3))

if __name__ == "__main__":
    main()