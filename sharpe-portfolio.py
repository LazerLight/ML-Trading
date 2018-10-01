#Daily Portfolio Value

#Prices are first normalized: prices of every day is divided by the start date value
#               then alloced: normalized prices * allocation percentage of the stock in the portfolio
#                   pos_vals: alloced * start_val (say $1 000 000)
#                   port_val: sum the pos_vals per row to give the value of the portfolio on each date.
# You can then calculate daily returns. Since d.r. of day 1 is 0, bypass it with daily_returns = daily_returns[1:]

#Statistics of interest for a protfolio's daily returns:
# cumulative returns: cum_ret = (port_val[-1] / port_val[0]) - 1
# avg_daily_returns: daily_returns.mean()
# std_daily_returns: daily_returns.std()

#sharpe_ratio: (Rp-Rf)/op
#Rp: Portfolio r.o.r, Rf: Risk-free r.o.r, op: std dev of portfolio r.o.r (volatility)
#Ex-ante Sharpe Ratio: 
# (mean Rp- mean Rf)) / (std*(Rp-Rf))
#In Python, this simplifies to mean*(daily_returns - daily_riskfree)/ (std * (daily_returns))

#Adjustment Raio:
#S.R. is an annual measure, frequency of sampling affects it. Adjust with S.R.annualized = K*S.R.
#Where k = sq.rt(samples per year). So sampling daily would be 252 days (the number of trading days in a year)
 