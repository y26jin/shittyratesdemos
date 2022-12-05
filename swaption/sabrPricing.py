#%%
# European Swaption
import QuantLib as ql
import pandas as pd

'''
Step 1: Set up basics
'''

todaysDate = ql.Date(21, ql.November, 2022)
ql.Settings.instance().evaluationDate = todaysDate
calendar = ql.TARGET()
settlementDate = ql.Date(25, ql.November, 2022)

# Market data

swaptionVols = [
    # maturity,          length,             volatility
    (ql.Period(1, ql.Years), ql.Period(5, ql.Years), 0.1148),
    (ql.Period(2, ql.Years), ql.Period(4, ql.Years), 0.1108),
    (ql.Period(3, ql.Years), ql.Period(3, ql.Years), 0.1070),
    (ql.Period(4, ql.Years), ql.Period(2, ql.Years), 0.1021),
    (ql.Period(5, ql.Years), ql.Period(1, ql.Years), 0.1000),
]

def calibrate(model, helpers, l, name):
    print(f"Model: {name}")

    method = ql.Simplex(l)
    model.calibrate(helpers, method, ql.EndCriteria(1000, 250, 1e-7, 1e-7, 1e-7))

    print("Parameters: %s" % model.params())

    totalError = .0
    data = []
    for swaption, helper in zip(swaptionVols, helpers):
        maturity, length, vol = swaption
        NPV = helper.modelValue()
        #implied = helper.impliedVolatility(NPV, 1.0e-4, 1000, 0.05, 0.50)
        # Note: The SABR parameters were not optimized!
        implied = ql.sabrVolatility(NPV, 1000.0, maturity.length(), 0.1, .1, .1, .1)
        error = implied - vol
        totalError += abs(error)
        data.append((maturity, length, vol, implied, error))
    averageError = totalError/len(helpers)
    
    print(pd.DataFrame(data, columns=["maturity", "length", "volatility", "implied", "error"]))
    print("Average error: %.4f" % averageError)


# %%
'''
Step 2: Define underlying vanilla swap
'''

rate = ql.QuoteHandle(ql.SimpleQuote(0.04875825))
termStructure = ql.YieldTermStructureHandle(ql.FlatForward(settlementDate, rate, ql.Actual365Fixed()))

# Define ATM/OTM/ITM swaps

swapEngine = ql.DiscountingSwapEngine(termStructure)

fixedLegFrequency = ql.Annual
fixedLegTenor = ql.Period(1, ql.Years)
fixedLegConvention = ql.Unadjusted
floatingLegConvention = ql.ModifiedFollowing
fixedLegDayCounter = ql.Thirty360(ql.Thirty360.European)
floatingLegFrequency = ql.Semiannual
floatingLegTenor = ql.Period(6, ql.Months)

payFixed = ql.Swap.Payer
fixingDays = 2
index = ql.Euribor6M(termStructure)
floatingLegDayCounter = index.dayCounter()

swapStart = calendar.advance(settlementDate, 1, ql.Years, floatingLegConvention)
swapEnd = calendar.advance(swapStart, 5, ql.Years, floatingLegConvention)

fixedSchedule = ql.Schedule(
    swapStart,
    swapEnd,
    fixedLegTenor,
    calendar,
    fixedLegConvention,
    fixedLegConvention,
    ql.DateGeneration.Forward,
    False,
)

floatingSchedule = ql.Schedule(
    swapStart,
    swapEnd,
    floatingLegTenor,
    calendar,
    floatingLegConvention,
    floatingLegConvention,
    ql.DateGeneration.Forward,
    False,
)

dummy = ql.VanillaSwap(
    payFixed, 100.0, fixedSchedule, .0, fixedLegDayCounter, floatingSchedule, index, .0, floatingLegDayCounter
)
dummy.setPricingEngine(swapEngine)
atmRate = dummy.fairRate()

atmSwap = ql.VanillaSwap(
    payFixed, 1000.0, fixedSchedule, atmRate, fixedLegDayCounter, floatingSchedule, index, .0, floatingLegDayCounter
)
otmSwap = ql.VanillaSwap(
    payFixed, 1000.0, fixedSchedule, atmRate * 1.2, fixedLegDayCounter, floatingSchedule, index, .0, floatingLegDayCounter
)
itmSwap = ql.VanillaSwap(
    payFixed, 1000.0, fixedSchedule, atmRate * 0.8, fixedLegDayCounter, floatingSchedule, index, .0, floatingLegDayCounter
)
atmSwap.setPricingEngine(swapEngine)
otmSwap.setPricingEngine(swapEngine)
itmSwap.setPricingEngine(swapEngine)

helpers = [
    ql.SwaptionHelper(
        maturity,
        length,
        ql.QuoteHandle(ql.SimpleQuote(vol)),
        index,
        index.tenor(),
        index.dayCounter(),
        index.dayCounter(),
        termStructure,
    )
    for maturity, length, vol in swaptionVols
]

times = {}
for h in helpers:
    for t in h.times():
        times[t] = 1
times = sorted(times.keys())

grid = ql.TimeGrid(times, 30)

G2model = ql.G2(termStructure)
HWmodel = ql.HullWhite(termStructure)
BKmodel = ql.BlackKarasinski(termStructure)

# Calibration

# G2 model
for h in helpers:
    h.setPricingEngine(ql.G2SwaptionEngine(G2model, 6.0, 16))
calibrate(G2model, helpers, 0.05, "G2 analytic formulae")

# Hull White model
for h in helpers:
    h.setPricingEngine(ql.JamshidianSwaptionEngine(HWmodel))
calibrate(HWmodel, helpers, 0.05, "Hull-White analytic formulae")

# Black-Karasinski model
for h in helpers:
    h.setPricingEngine(ql.TreeSwaptionEngine(BKmodel, grid))
calibrate(BKmodel, helpers, 0.05, "Black-Karasinski numerical calibration")
# %%
'''
Step 3: Price European Swaptions
'''
print("European Exercise: ")
exerciseDate = calendar.advance(todaysDate, ql.Period('5y'))
exercise = ql.EuropeanExercise(exerciseDate)

atmSwaption = ql.Swaption(atmSwap, exercise)
otmSwaption = ql.Swaption(otmSwap, exercise)
itmSwaption = ql.Swaption(itmSwap, exercise)

data = []

# 1. G2 Model Swaption
atmSwaption.setPricingEngine(ql.TreeSwaptionEngine(G2model, 50))
otmSwaption.setPricingEngine(ql.TreeSwaptionEngine(G2model, 50))
itmSwaption.setPricingEngine(ql.TreeSwaptionEngine(G2model, 50))

data.append(("G2 analytic", itmSwaption.NPV(), atmSwaption.NPV(), otmSwaption.NPV()))

# 2. Hull-White Model Swaption
atmSwaption.setPricingEngine(ql.TreeSwaptionEngine(HWmodel, 50))
otmSwaption.setPricingEngine(ql.TreeSwaptionEngine(HWmodel, 50))
itmSwaption.setPricingEngine(ql.TreeSwaptionEngine(HWmodel, 50))

data.append(("HW analytic", itmSwaption.NPV(), atmSwaption.NPV(), otmSwaption.NPV()))

# 3. Black-Karasinski Model Swaption
atmSwaption.setPricingEngine(ql.TreeSwaptionEngine(BKmodel, 50))
otmSwaption.setPricingEngine(ql.TreeSwaptionEngine(BKmodel, 50))
itmSwaption.setPricingEngine(ql.TreeSwaptionEngine(BKmodel, 50))

data.append(("BK numerical", itmSwaption.NPV(), atmSwaption.NPV(), otmSwaption.NPV()))

print(pd.DataFrame(data, columns=["model", "in-the-money", "at-the-money", "out-of-the-money"]))
# %%
'''
Step Bonus: Price Bermudan Swaptions(aka, exercise at set dates)
'''
print("Bermudan Exercise:")
bermudanDates = [d for d in fixedSchedule][:-1]
exercise2 = ql.BermudanExercise(bermudanDates)

atmSwaption2 = ql.Swaption(atmSwap, exercise2)
otmSwaption2 = ql.Swaption(otmSwap, exercise2)
itmSwaption2 = ql.Swaption(itmSwap, exercise2)

data2 = []

# +
atmSwaption2.setPricingEngine(ql.TreeSwaptionEngine(G2model, 50))
otmSwaption2.setPricingEngine(ql.TreeSwaptionEngine(G2model, 50))
itmSwaption2.setPricingEngine(ql.TreeSwaptionEngine(G2model, 50))

data2.append(("G2 analytic", itmSwaption2.NPV(), atmSwaption2.NPV(), otmSwaption2.NPV()))

# +
atmSwaption2.setPricingEngine(ql.TreeSwaptionEngine(HWmodel, 50))
otmSwaption2.setPricingEngine(ql.TreeSwaptionEngine(HWmodel, 50))
itmSwaption2.setPricingEngine(ql.TreeSwaptionEngine(HWmodel, 50))

data2.append(("HW analytic", itmSwaption2.NPV(), atmSwaption2.NPV(), otmSwaption2.NPV()))
# +
atmSwaption2.setPricingEngine(ql.TreeSwaptionEngine(BKmodel, 50))
otmSwaption2.setPricingEngine(ql.TreeSwaptionEngine(BKmodel, 50))
itmSwaption2.setPricingEngine(ql.TreeSwaptionEngine(BKmodel, 50))

data2.append(("BK numerical", itmSwaption2.NPV(), atmSwaption2.NPV(), otmSwaption2.NPV()))
# -

print(pd.DataFrame(data2, columns=["model", "in-the-money", "at-the-money", "out-of-the-money"]))
# %%
