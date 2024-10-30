import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
import matplotlib.pylab as plt

# Implied Volatility using SABR Model
def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    # if K is at-the-money-forward
    if abs(F - K) < 1e-12:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
        sabrsigma = VolAtm
    else:
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*np.log(F/X)
        zhi = np.log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(np.log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((np.log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom
    return sabrsigma

beta = 0.7 

def sabrcalibration(x, strikes, vols, F, T): # Vector, x = (alpha, rho, nu)
    err = 0.0
    for i, vol in enumerate(vols):
        err += (vol - SABR(F, strikes[i], T, x[0], beta, x[1], x[2]))**2
    return err

# Convert from Option Price to Implied Vol
def impliedVolatility(F0, K, r, price, T, payoff, beta):
    try:
        if (payoff.lower() == 'call'):
            impliedVol = brentq(lambda x: price -
                                DisplacedDiffusionCall(F0, K, r, x, T, beta),
                                1e-12, 10.0)
        elif (payoff.lower() == 'put'):
            impliedVol = brentq(lambda x: price -
                                DisplacedDiffusionPut(F0, K, r, x, T, beta),
                                1e-12, 10.0)
        else:
            raise NameError('Payoff type not recognized')
    except Exception:
        impliedVol = np.nan
    return impliedVol

# Black Model & Displaced-Diffusion Model for C+P Prices
def BlackLognormalCall(F0, K, r, sigma, T):
    d1 = (np.log(F0/K)+0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*(F0*norm.cdf(d1)-K*norm.cdf(d2))
def BlackLognormalPut(F0, K, r, sigma, T):
    d1 = (np.log(F0/K)+0.5*sigma**2*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*(K*norm.cdf(-d2)-F0*norm.cdf(-d1))

def DisplacedDiffusionCall(S, K, r, sigma, T, beta):
    return BlackLognormalCall(S/beta, K+((1-beta)/beta)*F0, r, sigma*beta, T)
def DisplacedDiffusionPut(S, K, r, sigma, T, beta):
    return BlackLognormalPut(S/beta, K+((1-beta)/beta)*F0, r, sigma*beta, T)

def ddcalibration(x, F0, strikes, r, market_prices, T):
    sigma, beta = x
    err = 0.0
    for i, K in enumerate(strikes):
        model_price = DisplacedDiffusionCall(F0, K, r, sigma, T, 0.7)
        err += (market_prices[i] - model_price)**2  # squared error
    return err
#---------------------------------------------------------------------------------------------------#
# Market Options Data (SPX/SPY)
df = SPY_df
df['mid'] = 0.5*(df['best_bid'] + df['best_offer'])
df['strike'] = df['strike_price']*0.001
df['payoff'] = df['cp_flag'].map(lambda x: 'call' if x == 'C' else 'put')
exdate = sorted(df['exdate'].unique())[0]
df = df[df['exdate'] == exdate]
days_to_expiry = (pd.Timestamp(str(exdate)) - pd.Timestamp('2020-12-01')).days

def interpolate_r(days_to_expiry):
    r = interp1d(Rates_df['days'], Rates_df['rate'], kind='linear', fill_value="extrapolate")
    return r(days_to_expiry)/100

T = days_to_expiry/365
S = 3662.45
r = interpolate_r(days_to_expiry)
F0 = S*np.exp(r*T)

# Calculation of Implied Volatility for C+P 
df['vols'] = df.apply(lambda x: impliedVolatility(F0, x['strike'], r, x['mid'], T, x['payoff'], beta), axis=1)
df.dropna(inplace=True)

call_df = df[df['payoff'] == 'call']
put_df = df[df['payoff'] == 'put']
strikes = put_df['strike'].values
impliedvols = []
for K in strikes:    
    if K > F0:
        impliedvols.append(call_df[call_df['strike'] == K]['vols'].values[0])
    else:
        impliedvols.append(put_df[put_df['strike'] == K]['vols'].values[0])

df = pd.DataFrame({'strike': strikes, 'impliedvol': impliedvols})
# DisplacedDiffusion Calibration
initial_guess_dd = [0.2, 0.5]
result_dd = least_squares(lambda x: ddcalibration(x, F0, df['strike'].values, df['mid'].values, r, T), initial_guess_dd)
sigma_dd, beta_dd = result_dd.x
#---------------------------------------------------------------------------------------------------#
# SABR Calibration
df = pd.DataFrame({'strike': strikes, 'impliedvol': impliedvols})

initialGuess = [0.02, 0.2, 0.1] # [alpha, rho, nu]
res = least_squares(lambda x: sabrcalibration(x, df['strike'], df['impliedvol'], F, T), initialGuess)

alpha = res.x[0]
rho = res.x[1]
nu = res.x[2]

sabrvols = []
for K in strikes:
    sabrvols.append(SABR(F, K, T, alpha, beta, rho, nu))

plt.figure(tight_layout=True)
plt.plot(strikes, df['impliedvol'], 'gs', label='Market Vols')
plt.plot(strikes, sabrvols, 'm--', label='SABR Vols')
plt.legend()
plt.show()

# Print calibrated parameters
print(f'Displaced-Diffusion Model: beta = {beta_dd}')
print(f'Displaced-Diffusion Model: sigma = {sigma_dd}')
print(f'SABR Model: alpha = {alpha:.3f}, rho = {rho:.3f}, nu = {nu:.3f}')
