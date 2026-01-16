import logging
import os
import sys
import pandas as pd
import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

# https://www.ote-cr.cz/cs/statistika/rocni-zprava?date=2019-01-01

def get_years(start_date, end_date):

    # get list of years to download
    years = list(range(start_date.year, end_date.year + 1))
    years = [str(year) for year in years]
    return years

def get_ote_url(year):
    # https://www.ote-cr.cz/pubweb/attachments/62_162/2019/Rocni_zprava_o_trhu_2019_V2_plyn.zip
    # https://www.ote-cr.cz/pubweb/attachments/62_162/2024/Rocni_zprava_o_trhu_2024_V2_plyn.zip
    url = f'https://www.ote-cr.cz/pubweb/attachments/62_162/{year}/' 
    url += f'Rocni_zprava_o_trhu_{year}_V2_plyn.zip'
    return url

def download_zipfile_and_create_df(year):
    # try:        
        url = get_ote_url(year)
        resp = urlopen(url)
        ZipFile(BytesIO(resp.read())).extractall('./extract')

        # get data from sheets
        if year < '2024': # migh there be exception for year 2024 and higher
            filename_xlsx = f'./extract/Rocni_zprava_o_trhu_plyn_{year}_V2.xls'
            odchylky_nc_bal = pd.read_excel(filename_xlsx, sheet_name='Odchylky NC BAL', header=1, skiprows=4) # Trh s NFL +
        else:
            filename_xlsx = f'./extract/Rocni_zprava_o_trhu_plyn_{year}_V2.xlsx'
            odchylky_nc_bal = pd.read_excel(filename_xlsx, sheet_name='Odchylky NC BAL', header=1, skiprows=4) # Trh s NFL +

        # os.remove(filename_xlsx)
        return odchylky_nc_bal

    # except Exception as e:
    #     logging.error('ERROR: FAILED TO DOWNLOAD ZIPFILE AND CREATE DF. REASON: %s' % e)

def get_diagrams(start_date, end_date):

    years = get_years(start_date, end_date)
    df_odchylky_nc_bal = pd.DataFrame()
    for year in years:
        odchylky_nc_bal = download_zipfile_and_create_df(year)
        # if int(year) >= 2019: # inconsistency in data
        #     df = df.drop(index=df.index[-4:])
        logging.info('DATA FOR YEAR ' + year + ' DOWNLOADED')
        df_odchylky_nc_bal = pd.concat([df_odchylky_nc_bal, odchylky_nc_bal])
    return df_odchylky_nc_bal

def arrange_df_odchylky_nc_bal(df, date_from, date_to):
    df.rename(columns={
        'Plynárenský den': 'DATE',
        'Systémová odchylka\n(MWh)' : 'systemova_odchylka',
        'Kladné odchylky\n(MWh)'    : 'kladne_odchylky',
        'Záporné odchylky\n(MWh)'   : 'zaporne_odchylky',
        'Přetoky z PS do DS\n(MWh)' : 'pretoky_ps_ds',
        'Kurz ČNB\n(Kč/EUR)'        : 'kurz_cnb',
        'Použitelná cena pro kladné vyrovnávací množství \n(Kč/MWh)'    : 'cena_kladne_kc',
        'Použitelná cena pro záporné vyrovnávací množství \n(Kč/MWh)'   : 'cena_zaporne_kc',
        'Použitelná cena pro kladné vyrovnávací množství \n(EUR/MWh)'   : 'cena_kladne_eur',
        'Použitelná cena pro záporné vyrovnávací množství\n(EUR/MWh)'   : 'cena_zaporne_eur',
        'Index OTE\n(EUR/MWh)'      : 'index_ote',
        'Referenční cena NCG\n(EUR/MWh)'                                : 'referencni_cena_ncg',
        'Měsíční vyrovnávací cena\n(Kč/MWh)'                            : 'mesicni_vyrovnavaci_cena',
        'Vyrovnávací akce TSO celkem\n(MWh)'                            : 'vyrovnavaci_akce_tso',
        'Vyrovnávací služba TSO celkem\n(MWh)'                          : 'vyrovnavaci_sluzba_tso',
        'Poskytovaná flexibilita prostřednictvím akumulace\n(MWh)'      : 'flexibilita_akumulace',
        'Součet denních vyrovnávacích množství SZ/ZÚ\n(MWh)'            : 'soucet_vyrovnavacich_mnozstvi',
        'Součet alokací využití flexibility SZ/ZÚ\n(MWh)'               : 'soucet_alokaci_flexibility',
        'Spotřeba ČR\n(MWh)'        : 'spotreba_cr',
        'Úroveň flexibility'        : 'uroven_flexibility'
        }, inplace=True)

     # make sure that all years are the same
    df.loc[:, 'DATE'] = pd.to_datetime(df['DATE'], format='%d.%m.%Y').dt.date
    # df.to_csv('test.csv', index=True)
    df.set_index('DATE', inplace=True)
    # df = df.loc[df.index >= date_from.normalize().date()].dropna()
    # df = df.loc[df.index <= date_to.normalize().date()].dropna()
    return df

def write_to_output(df, table):
    """
    Write output to database
    Args:
        df: Output dataframe
        table: Table name
    """
    try:
        output_data_folder = 'output_data'
        if not os.path.exists(output_data_folder):
            os.makedirs(output_data_folder)
        df.to_csv(f"{output_data_folder}/{table}.csv", index=True)
        logging.info(table.upper() + ': SUCCESSFUL WRITE TO CSV.')
    except Exception as e:
        logging.error(table.upper() + ': FAILED TO SAVE TO CSV. REASON: %s' % e)

# def write_to_pickle(df):
#     PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
#     pickle_name = f'{PROJECT_ROOT}/pickled_data/temperatures.pickle'

#     try:
#         df.to_pickle(pickle_name)
#         logging.info('PICKLE write successful: ' + pickle_name)
#     except Exception as e:
#         logging.error(e)

def run_ote_report(date_from, date_to, table):
    df_odchylky_nc_bal = get_diagrams(date_from, date_to)
    
    df = arrange_df_odchylky_nc_bal(df_odchylky_nc_bal, date_from, date_to)

    logging.info(f'DATA FROM {date_from.date()} TO {date_to.date()} READY TO STORE')
    logging.info('\n' + str(df))
    write_to_output(df, table)

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S",
                        format='%(asctime)s %(levelname)s | %(message)s')

    # date_from = pd.to_datetime('today').normalize() - timedelta(days=1)  # or pd.to_datetime('2023-02-17')
    # date_to = pd.to_datetime('today')
    date_from = pd.to_datetime('2016-01-01') # below 2016 problems
    date_to = pd.to_datetime('2025-12-31')

    table = 'OTE_NG_ODCHYLKY_NC_BAL'

    # execution
    # try:
    run_ote_report(date_from, date_to, table)
    # except Exception as e:
    #     logging.error('ERROR: FAILED TO EXECUTE THE SCRIPT. REASON: %s' % e)
