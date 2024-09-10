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
import matplotlib.pyplot as plt

'''
Importation des données
'''

df = pd.read_excel('output_dataset.xlsx', engine='openpyxl')


'''
Fonctions auxiliaires
'''

def CDF_norm01(x):
    return norm.cdf(x)

def discount_factor(interest_rate, maturity):
    return exp(-interest_rate * maturity)

def d1(firm_value, debt, interest_rate, firm_volatility, maturity):
    return (log(firm_value / debt) + (interest_rate + 0.5 * firm_volatility**2) * maturity) / (firm_volatility * sqrt(maturity))

def d2(firm_value, debt, interest_rate, firm_volatility, maturity):
    return d1(firm_value, debt, interest_rate, firm_volatility, maturity) - firm_volatility * sqrt(maturity)

def equity_option(firm_value, firm_volatility, debt, interest_rate, maturity):
    d1_val = d1(firm_value, debt, interest_rate, firm_volatility, maturity)
    d2_val = d2(firm_value, debt, interest_rate, firm_volatility, maturity)
    return firm_value * CDF_norm01(d1_val) - debt * discount_factor(interest_rate, maturity) * CDF_norm01(d2_val)

def equation1(firm_value, firm_volatility, debt, interest_rate, maturity, equity_value):
    return equity_option(firm_value, firm_volatility, debt, interest_rate, maturity) - equity_value

def equation2(firm_value, firm_volatility, debt, interest_rate, maturity, equity_value, equity_volatility):
    d1_val = d1(firm_value, debt, interest_rate, firm_volatility, maturity)
    gauche = equity_volatility * equity_value
    droite = firm_value * firm_volatility * CDF_norm01(d1_val)
    return gauche - droite

def equations_system(debt, interest_rate, maturity, equity_value, equity_volatility):
    return lambda x: equation1(x[0], x[1], debt, interest_rate, maturity, equity_value), \
           lambda x: equation2(x[0], x[1], debt, interest_rate, maturity, equity_value, equity_volatility)

def solve_fsolve(init_cond, debt, interest_rate, maturity, equity_value, equity_volatility):
    eq_1, eq_2 = equations_system(debt, interest_rate, maturity, equity_value, equity_volatility)
    objective_func = lambda x: [eq_1(x), eq_2(x)]
    res = fsolve(objective_func, init_cond, full_output=True)
    return res

def PD(firm_value, debt, interest_rate, firm_volatility, maturity):
    return CDF_norm01(-d2(firm_value, debt, interest_rate, firm_volatility, maturity))


'''
Calcul de la PD
'''

def calculate_pd(row):
    debt = row['Valeurs dettes']
    equity_value = row['Valeurs Actions']
    equity_volatility = row['Volatilité']
    interest_rate = row['Taux sans risque']
    maturity = row['Maturité']
    
    init_condition = [debt + equity_value, equity_volatility] 
    
    # Résoudre pour obtenir la valeur de l'entreprise et sa volatilité
    firm_value, firm_volatility = solve_fsolve(init_condition, debt, interest_rate, maturity, equity_value, equity_volatility)[0]
    
    # Calculer la probabilité de défaut
    pd = PD(firm_value, debt, interest_rate, firm_volatility, maturity) * 100
    
    return pd

# Étape 3 : Appliquer la fonction à chaque ligne du DataFrame
df['PD (en %)'] = df.apply(calculate_pd, axis=1)

# Afficher le DataFrame avec la colonne PD ajoutée
print(df)

# Tracer l'évolution de la PD au cours du temps
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['PD (en %)'], linestyle='-', color='b')

# Ajouter des labels et un titre
plt.xlabel('Date')
plt.ylabel('Probabilité de Défaut (PD) en %')
plt.title('Évolution de la Probabilité de Défaut AAPL au cours du temps')
plt.grid(True)

# Afficher le graphique
plt.show()

# Tracer l'évolution de la PD au cours du temps
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['GDP'], linestyle='-', color='b')

# Ajouter des labels et un titre
plt.xlabel('Date')
plt.ylabel('GDP')
plt.title('Évolution du PIB au cours du temps')
plt.grid(True)

# Afficher le graphique
plt.show()




