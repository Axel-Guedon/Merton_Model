'''
Importation des modules
'''

import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import fsolve
import pandas as pd
from urllib.request import urlopen
import requests
import json
import yfinance as yf
import numpy as np


'''
Valeurs des actions et de la dette
Données AAPL annuelles du 30/09/2013 au 30/09/2023. (clé totalShareholderEquity et shortLongTermDebtTotal) (récupérées sur API Alpha Vantage)
'''

def get_equity_values(ticker, api_key) :
    
    url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()
    data = data['annualReports']
    i = 0
    list_equity_values = []
    list_debt_values = []
    list_year = []
    for element in data :
        list_equity_values.append(int(element['totalShareholderEquity']))
        debt_value = element['shortLongTermDebtTotal']
        if debt_value != 'None' :
            list_debt_values.append(int(debt_value))
        else:
            list_debt_values.append(0) 
        list_year.append(element['fiscalDateEnding'])
        i += 1
    return list_equity_values, list_debt_values, list_year


equity_values, debt_values, year_values = get_equity_values("AAPL", "N8IEZU3ABO0O1UX4")


'''
Transformation des données bilancielles en dataframe et ajout de la maturité
'''

df_bilan = pd.DataFrame({
    'Année': year_values,
    'Valeurs Actions': equity_values,
    'Valeurs dettes' : debt_values,
})

df_bilan = df_bilan[df_bilan['Valeurs dettes'] != 0]

print(df_bilan)

'''
Taux sans risque et dates 
Rendements journaliers des T-bond de maturité 1 an du 30/09/2013 au 29/09/2023
'''

df_rate = pd.read_excel("data_risk_free_rate.xlsx", sheet_name="Sheet1", usecols="E", skiprows=4113, nrows=2610)
df_rate = df_rate.rename(columns={df_rate.columns[0]: 'Taux sans risque'})



df_days = pd.read_excel("data_risk_free_rate.xlsx", sheet_name="Sheet1", usecols="A", skiprows=4113, nrows=2610)
df_days = df_days.rename(columns={df_days.columns[0]: 'Date'})
df_days['Date'] = pd.to_datetime(df_days['Date'])

df_rate = pd.concat([df_days, df_rate], axis=1)

print(df_rate)


'''
Volatilité historique obtenues via Yahoo Finance (calcul du 30/09/2013 au 29/09/2023)
'''


def compute_histo_vol(data, start_date, end_date):
    
    # Extraire les données pour la période spécifiée
    data_period = data.loc[start_date:end_date]
    
    # Calculer les rendements logarithmiques quotidiens
    log_returns = np.log(data_period['Adj Close'] / data_period['Adj Close'].shift(1))
    
    # Calculer la volatilité (écart-type des rendements logarithmiques) annualisée
    vol = np.std(log_returns) * np.sqrt(252)  # 252 correspond au nombre de jours de trading en une année
    
    return vol


def get_histo_vol(ticker, start_period, end_period) : 
     
    data = yf.download(ticker, start = start_period, end = end_period)
    volatilites_historiques = []
    window_size = 252

    for i in range (window_size, len(data)) :
        end_date = data.index[i]
        start_date = data.index[i-window_size]
        vol = compute_histo_vol(data, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        volatilites_historiques.append((end_date, vol))
    
    df_volatilities = pd.DataFrame(volatilites_historiques, columns=['Date', 'Volatilité'])
    
    return df_volatilities

df_equity_volatilities = get_histo_vol("AAPL", "2012-09-26", "2023-09-30")

print(df_equity_volatilities)

'''
Regroupement des données et substitution des missing values
'''

df_merged = pd.merge(df_equity_volatilities, df_rate, on='Date', how='inner') # Jointure sur les dates

def replace_nd_with_mean(df, column):

    for i in range(1, len(df) - 1):
        if df.loc[i, column] == "ND":

            # Remplacer "ND" par la moyenne des valeurs adjacentes
            prev_value = df.loc[i - 1, column]
            next_value = df.loc[i + 1, column]
            
            # Vérifier que les valeurs adjacentes ne sont pas "ND"
            if prev_value != "ND" and next_value != "ND":
                df.loc[i, column] = (float(prev_value) + float(next_value)) / 2
                
    # Si "ND" est au début ou à la fin, il n'y a pas de valeurs adjacentes pour calculer la moyenne
    # Cela doit être traité séparément si nécessaire

    return df

df_merged = replace_nd_with_mean(df_merged, "Taux sans risque")


'''
Affectition des bonnes données bilancielles à chaque ligne (vol, taux sans risque)
'''

df_bilan['Start_Date'] = pd.to_datetime(df_bilan['Année'])  # 30 septembre de l'année n
df_bilan['End_Date'] = df_bilan['Start_Date'] + pd.DateOffset(years=1) - pd.DateOffset(days=1)  # 29 septembre de l'année n+1


# Faire correspondre les valeurs d'actions et de dettes
def assign_bilan_data(date):
    bilan_row = df_bilan[(df_bilan['Start_Date'] <= date) & (df_bilan['End_Date'] >= date)]
    if not bilan_row.empty:
        return bilan_row.iloc[0]['Valeurs Actions'], bilan_row.iloc[0]['Valeurs dettes']
    else:
        return None, None

df_merged['Valeurs Actions'], df_merged['Valeurs dettes'] = zip(*df_merged['Date'].apply(assign_bilan_data))
df_merged['Année'] = df_merged['Date'].dt.year
print(df_merged)


"""
PIB USA de 2013 à 2023 en USD
"""

df = pd.read_excel('GDP_data.xlsx', engine='openpyxl')

year_data = df.columns
gdp_data = df.iloc[187, :].values

# Create the dataframe
gdp_df = pd.DataFrame({
    'Année': year_data,
    'GDP': gdp_data
})

# Supprimer la première ligne d'en-tête incorrecte
gdp_df = gdp_df.drop(0).reset_index(drop=True)

# Filtrer le DataFrame pour les années 2013 à 2023
filtered_df = gdp_df[(gdp_df['Année'] >= 2013) & (gdp_df['Année'] <= 2023)]

# Afficher le DataFrame filtré
print(filtered_df)

df_final = pd.merge(df_merged, filtered_df, on='Année', how='inner')
df_final['Maturité'] = 1
df_final = df_final.drop(columns=['Année'])
df_final['Taux sans risque'] = df_final['Taux sans risque'] / 100
df_final['GDP'] = df_final['GDP'] * 1000000000

print(df_final)

# Exporter le DataFrame en fichier Excel
df_final.to_excel('output_dataset.xlsx', index=False, engine='openpyxl')