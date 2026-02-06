import pandas as pd
import numpy as np

# Cesta k souboru
file_path = 'output_graph/weights_plots/weights_history_frank_wolfe_6_weeks_weighted.csv'

def generate_latex_table(file_path):
    try:
        # Načtení dat
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Soubor nebyl nalezen: {file_path}")
        return

    # Odstranění nenumerických sloupců (např. datum), pokud existují
    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int']).columns
    # Případně odstranění sloupce 'Unnamed: 0', pokud je v datech
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Výpočet průměrných vah a seřazení
    avg_weights = df.mean().sort_values(ascending=False)

    # Definice rodin modelů (Model Families)
    # Upravte tento slovník podle přesných názvů vašich modelů
    family_map = {
        'N-HiTS': 'Deep Learning',
        'TFT': 'Deep Learning',
        'LSTM': 'Deep Learning',
        'NeuralProphet': 'Deep Learning',
        'Transformer': 'Deep Learning',
        
        'CatBoost': 'Tree-Based',
        'XGBoost': 'Tree-Based',
        'LightGBM': 'Tree-Based',
        'ExtraTrees': 'Tree-Based',
        'RandomForest': 'Tree-Based',
        
        'SARIMAX': 'Statistical/Other',
        'Prophet': 'Statistical/Other',
        'AR_Eco': 'Statistical/Other',
        'KNN': 'Statistical/Other',
        'SVR': 'Statistical/Other',
        'ElasticNet': 'Statistical/Other',
        'BayesianRidge': 'Statistical/Other',
        'Lasso': 'Statistical/Other',
        'Ridge': 'Statistical/Other'
    }

    # Vytvoření souhrnného DataFrame
    summary = pd.DataFrame({'Base Learner': avg_weights.index, 'Weight': avg_weights.values})
    
    # Přiřazení rodiny, defaultně 'Statistical/Other' pokud není v mapě
    summary['Model Family'] = summary['Base Learner'].map(family_map).fillna('Statistical/Other')

    # Agregace modelů s malou vahou (< 1%)
    threshold = 0.01
    main_models = summary[summary['Weight'] >= threshold].copy()
    others = summary[summary['Weight'] < threshold]

    if not others.empty:
        others_weight = others['Weight'].sum()
        # Přidání řádku 'Others'. Obvykle se řadí pod 'Statistical/Other'
        others_row = pd.DataFrame({
            'Base Learner': ['Others (aggregated)'], 
            'Weight': [others_weight],
            'Model Family': ['Statistical/Other'] 
        })
        main_models = pd.concat([main_models, others_row], ignore_index=True)

    # Formátování váhy na procenta
    main_models['Avg. Weight (%)'] = (main_models['Weight'] * 100).map('{:.1f}\%'.format)

    # Definice pořadí rodin pro výpis
    family_order = ['Deep Learning', 'Tree-Based', 'Statistical/Other']
    
    # Pomocné sloupce pro třídění
    main_models['Family_Rank'] = main_models['Model Family'].map({v: i for i, v in enumerate(family_order)})
    main_models['Sort_Weight'] = main_models['Weight']
    # Zajistíme, aby "Others" bylo vždy na konci své skupiny
    main_models.loc[main_models['Base Learner'] == 'Others (aggregated)', 'Sort_Weight'] = -1

    # Seřazení: Nejdřív podle rodiny, pak podle váhy sestupně
    main_models = main_models.sort_values(by=['Family_Rank', 'Sort_Weight'], ascending=[True, False])

    # Generování LaTeX kódu
    print(r"\begin{table}[H]")
    print(r"\caption{Average weight allocation ($\bar{w}$) assigned by the Frank-Wolfe optimizer (Recency-Weighted strategy). Models with negligible weights ($< 1\%$) are aggregated.}")
    print(r"\label{tab:weights_summary}")
    print(r"\centering")
    print(r"\begin{tabular}{llc}")
    print(r"\toprule")
    print(r"\textbf{Model Family} & \textbf{Base Learner} & \textbf{Avg. Weight (\%)} \\")
    print(r"\midrule")

    for i, family in enumerate(family_order):
        family_data = main_models[main_models['Model Family'] == family]
        if family_data.empty:
            continue
            
        n_rows = len(family_data)
        first_row = True
        
        for _, row in family_data.iterrows():
            learner = row['Base Learner']
            if learner == 'Others (aggregated)':
                learner = r"\textit{Others (aggregated)}"
            
            weight_str = row['Avg. Weight (%)']
            
            if first_row:
                print(f"\\multirow{{{n_rows}}}{{*}}{{{family}}} & {learner} & {weight_str} \\\\")
                first_row = False
            else:
                print(f" & {learner} & {weight_str} \\\\")
        
        # Oddělovací čára, pokud následuje další rodina
        if i < len(family_order) - 1 and not main_models[main_models['Model Family'] == family_order[i+1]].empty:
             print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

# Spuštění
if __name__ == "__main__":
    generate_latex_table(file_path)