## Value:

  Market Cap: Stock price * # of stocks available

  Book Value: Tangible assests - intagible assests - liabilities

  Intrinsic Value: Dividends / discount rate


========================================
# Technical Analysis
    Historical price and volume only

    Compute statistics called indicators

    Rules of Thumb for using:

        1-Individual indicators are weak, use combinations (3-5)
        2-Look for contrasts in indicators (stock vs stock or stock vs market)
        3-Use over short time periods

## Indicators
    -Momentum: price / (price n days earlier) - 1

    -Simple moving average: n-day window of average
    SMA[t] = price[t] / price[t-n:t].mean() - 1
        Can be a proxy for value
        Graph SMA and see if it deviates from stock graph greatly
        Can combine with momentum and check where price graph crosses SMA graph

    -Bollinger Band
    Look for the crossing of price graph 'inside' BB

## Normalization
    Normalize indicators so one doesn't dominate the other (ex. momentum vs PE ratio)

    Done with: value - mean / value.std()

===============================
# Dealing with data
    -open
    -high
    -low
    -close
    -vol
    Each can apply to minutes, hours days etc

=========================
#Supervised Regression Learning
    Supervised: Provide examples: given x, expect y
    Regression: Numerical Prediction
    Learning: Provided Data

##Techniques
    Linear regression (is parametric: i.e. it finds parameters)

    k nearest neighbor (KNN) (instance-based)