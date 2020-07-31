import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# taking some stocks to fit them on efficient frontier
hul=pd.read_csv("HINDUNILVR.csv")
icici=pd.read_csv('ICICIBANK.csv')
lt=pd.read_csv('LT.csv')
mgl=pd.read_csv('MGL.csv')
mutfin=pd.read_csv('MUTHOOTFIN.csv')
pidlite=pd.read_csv('PIDILITIND.csv')
relaxo=pd.read_csv('RELAXO.csv')
tcs=pd.read_csv('TCS.csv')
bajfin=pd.read_csv('BAJFINANCE.csv')
brit=pd.read_csv('BRITANNIA.csv')
hcl=pd.read_csv('HCLTECH.csv')
hdfcam=pd.read_csv('HDFCAMC.csv')
ap=pd.read_csv('ASIANPAINT.csv')
astral=pd.read_csv('ASTRAL.csv')
igl=pd.read_csv('IGL.csv')

X=pd.DataFrame()
name=[hul,icici,lt,mgl,mutfin,pidlite,relaxo,tcs,bajfin,brit,hcl,hdfcam,ap,astral,igl]
name2=['hul','icici','lt','mgl','mutfin','pidlite','relaxo','tcs','bajfin','brit','hcl','hdfcam','ap','astral','igl']

for i in range(len(name)):
    X[name2[i]]=name[i]['Close']
    
returns=X.pct_change()
returns=returns.iloc[1:,:]

import modules as md
cov=returns.cov()

rets=md.annualize_rets(returns,252)
weights=md.optimal_weights(20,rets, cov)
rets = [md.portfolio_return(w, rets) for w in weights]
vols = [md.portfolio_vol(w, cov) for w in weights]
ef = pd.DataFrame({
    "Returns": rets, 
    "Volatility": vols
})

ef.plot.line(x="Volatility", y="Returns", style=".-")

#print the weights corresponding to minimum volatility at 11% returns
md.minimize_vol(0.11, md.annualize_rets(returns,252), cov)

def goodlook(pr,er,cov,l1):
    l2=md.minimize_vol(pr,er, cov)*100
    dicts={'stocks':l1,'price':l2}
    return pd.DataFrame(dicts)

df=goodlook(0.11,md.annualize_rets(returns,252), cov,name2)

md.msr(0.03,md.annualize_rets(returns,252), cov)

# choosing five best stocks
l=['relaxo','hul','hdfcam','igl','bajfin']

rets=md.annualize_rets(returns[l],252)
weights=md.optimal_weights(20,rets, cov.loc[l,l])
rets = [md.portfolio_return(w, rets) for w in weights]
vols = [md.portfolio_vol(w, cov.loc[l,l]) for w in weights]
ef = pd.DataFrame({
    "Returns": rets, 
    "Volatility": vols
})

#plot efficient frontier
ef.plot.line(x="Volatility", y="Returns", style=".-")    

#weights of various stocks to get 20% returns
df3=goodlook(0.2,md.annualize_rets(returns[l],252), cov.loc[l,l],l)
    
