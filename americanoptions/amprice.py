#%%
import QuantLib as ql
import matplotlib.pyplot as plt

maturity_date = ql.Date(15, 1, 2023)
spot_price = 127.62
strike = 130
volatility = .20
dividend_rate = 0.0163
option_type = ql.Option.Call

risk_free_rate = .001
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()

calculation_date = ql.Date(8, 5, 2022)
ql.Settings.instance().evaluationDate = calculation_date

payoff = ql.PlainVanillaPayoff(option_type, strike)
settlement = calculation_date

am_exercise = ql.AmericanExercise(settlement, maturity_date)
american_option = ql.VanillaOption(payoff, am_exercise)

spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, risk_free_rate, day_count))
dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, dividend_rate, day_count))
flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(calculation_date, calendar, volatility, day_count))

bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_yield, flat_ts, flat_vol_ts)

steps = 200
binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", steps)
american_option.setPricingEngine(binomial_engine)

print(american_option.NPV())

def binomial_price(option, bsm_process, steps):
    binomal_engine = ql.BinomialVanillaEngine(bsm_process, "crr", steps)
    option.setPricingEngine(binomial_engine)
    return option.NPV()

steps = range(5, 200, 1)
am_prices = [binomial_price(american_option, bsm_process, step) for step in steps]
plt.plot(steps, am_prices, label="American Option", lw=2, alpha=.6)
plt.xlabel("Steps")
plt.ylabel("Price")
plt.ylim(6.7, 7)
plt.title("Binomial Tree Price for Varying Steps")
plt.legend()

# %%
