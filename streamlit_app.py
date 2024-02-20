import streamlit as st  # Importing the Streamlit library for creating web apps
import pandas as pd  # Importing pandas library for data manipulation
import yfinance as yf  # Importing yfinance for fetching stock data
from datetime import date, timedelta  # Importing necessary date-related modules
from statistics import mean, stdev  # Importing statistical functions for calculating mean and standard deviation
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
from tabulate import tabulate  # Importing tabulate for creating tables

# Defining a class for Stock
class Stock():
    # Initializing the Stock class with code and start date
    def __init__(self, code, start_date: date):
        self.code = code
        # Downloading stock data for the specified code and date range
        self.stock_data = yf.download(self.code, start=(start_date - timedelta(days=31)).strftime("%Y-%m-%d"))
        self.stock_data = self.stock_data['Adj Close']
        
    # Method to get current price for a given date
    def CurPrice(self, curDate: date):
        return float(self.stock_data[curDate.strftime("%Y-%m-%d")])

    # Method to calculate monthly return for a given date
    def MonthlyRet(self, curDate: date):
        last30dp_list = self.Last30daysPrice(curDate)
        if last30dp_list is not None and len(last30dp_list) > 0:
            return (float(last30dp_list[-1]) / float(last30dp_list[0])) - 1
        else:
            return 0
    # Method to calculate daily return for a given date
    def DailyRet(self, curDate: date):
        formatted_date = curDate.strftime("%Y-%m-%d")

        if formatted_date in self.stock_data.index:
            current_value = float(self.stock_data.loc[formatted_date])
            previous_value = float(self.stock_data.iloc[(len(self.stock_data[:curDate]) - 2)])

            return (current_value / previous_value) - 1
        else:
            return 0
    # Method to get last 30 days price for a given date
    def Last30daysPrice(self, curDate: date):
        if (curDate - timedelta(days=30)).strftime("%Y-%m-%d") in self.stock_data.index:
            return [float(i) for i in list(
                self.stock_data[(curDate - timedelta(days=30)).strftime("%Y-%m-%d"):curDate.strftime("%Y-%m-%d")])]
        

class PerformanceMetrics:
    def __init__(self, equity_curve):
        self.equity_curve = equity_curve

# Defining a class for Portfolio
class Portfolio():
    # Initializing the Portfolio class with start date, end date, initial equity, strategy duration, and list of stocks
    def __init__(self, start_date: date, end_date: date, initial_equity: float, strategy_duration: int, nifty50_stock):
        self.nifty50_stock = nifty50_stock
        self.start_date = start_date
        self.end_date = end_date
        self.initial_equity = initial_equity
        self.nifty50_stocklist = [Stock(i, start_date) for i in self.nifty50_stock]
        self.strategy_duration = strategy_duration
        self.benchmark, self.benchmark_DR = self.Benchmark()
        self.strategy, self.strategy_DR, self.strategy_CP = self.Strategy()
        self.selected_stocks = self.strategy_CP
        self.summarized_perf = self.calculatePerformance()

    # Method to select stocks for a given month
    def MonthlyStockSelection(self, curDate: date):
        selected_stocks = []
        for stock in self.nifty50_stocklist:
            if stock.MonthlyRet(curDate) > 0:
                selected_stocks.append(stock)
        return selected_stocks

    # Method for setting the benchmark performance
    def Benchmark(self):
        Nifty50 = Stock('^NSEI', self.start_date)
        benchmark = pd.DataFrame([self.initial_equity], index=[str(self.start_date.strftime("%Y-%m-%d"))],
                                columns=['Close']).astype(float)
        benchmark_DR = [0]
        for i, e in Nifty50.stock_data[(self.start_date + timedelta(days=1)).strftime("%Y-%m-%d"):].items():
            benchmark_DR.append(Nifty50.DailyRet(i))
            benchmark.loc[str(i.strftime("%Y-%m-%d"))] = benchmark.iloc[len(benchmark) - 1] * (1 + benchmark_DR[-1])
        return benchmark, benchmark_DR

    # Method to implement the strategy
    def Strategy(self):
        start_date = self.start_date
        strategy = pd.DataFrame([self.initial_equity], index=[str(self.start_date.strftime("%Y-%m-%d"))],
                                columns=['Close']).astype(float)
        strategy_DR = [0]
        current_portfolio = []
        while start_date < self.end_date:
            current_portfolio = self.MonthlyStockSelection(start_date)
            print(start_date)
            for i, e in self.benchmark[(start_date + timedelta(days=1)).strftime("%Y-%m-%d"):].iterrows():
                if date.fromisoformat(i) > (start_date + timedelta(days=self.strategy_duration)):
                    break
                daily_returns = self.calculateOverallDailyRet(current_portfolio, date.fromisoformat(i))
                strategy_DR.append(daily_returns)
                strategy.loc[i] = strategy.iloc[len(strategy) - 1] * (1 + strategy_DR[-1])
                end_date = date.fromisoformat(i)
            print(end_date)
            if start_date == end_date:
                break
            start_date = end_date
        return strategy, strategy_DR, [s.code for s in current_portfolio]
    
    # Method to calculate overall daily return
    def calculateOverallDailyRet(self, current_portfolio, curDate: date):
        if len(current_portfolio) > 0:
            sum_daily_return = 0
            for stock in current_portfolio:
                sum_daily_return += stock.DailyRet(curDate)
            return sum_daily_return / len(current_portfolio)
        else:
            return 0 

    # Method to calculate Compound Annual Growth Rate (CAGR)
    def calculate_cagr(self, equity_curve):
        initial_value = equity_curve.iloc[0]
        final_value = equity_curve.iloc[-1]

        if (initial_value == final_value).all():
            return 0

        t = (self.end_date - self.start_date).days / 365
        CAGR = (((final_value / initial_value) ** (1 / t)) - 1) * 100
        return CAGR
    
    # Method to calculate Sharpe Ratio
    def calculate_sharpe_ratio(self,equity_curve):
        SR = ((252 ** 0.5) * (mean(equity_curve) / stdev(equity_curve)))
        return SR

    # Method to calculate volatility
    def calculate_volatility(self,equity_curve):
        Volatility = ((252 ** 0.5) * stdev(equity_curve)) * 100
        return Volatility

    # Method to calculate maximum drawdown
    def calculate_max_drawdown(self, equity_curve: pd.DataFrame) -> float:
        peak_value = equity_curve['Close'].cummax()
        drawdown = (equity_curve['Close'] - peak_value) / peak_value
        max_drawdown = drawdown.min()
        return max_drawdown * 100  
    

    # Method to calculate drawdown volatility
    def calculate_drawdown_volatility(self, equity_curve: pd.DataFrame) -> float:
        peak_value = equity_curve['Close'].cummax()
        drawdown = (equity_curve['Close'] - peak_value) / peak_value
        drawdown_volatility = drawdown.std() * 100  
        return drawdown_volatility

    # Method to calculate performance metrics
    def calculatePerformance(self):
        benchmark_CAGR = self.calculate_cagr(self.benchmark)[0]
        strategy_CAGR = self.calculate_cagr(self.strategy)[0]
        benchmark_Volatility = self.calculate_volatility(self.benchmark_DR)
        strategy_Volatility = self.calculate_volatility(self.strategy_DR)
        benchmark_SR = self.calculate_sharpe_ratio(self.benchmark_DR)
        strategy_SR = self.calculate_sharpe_ratio(self.strategy_DR)

        benchmark_maxdd = self.calculate_max_drawdown(self.benchmark)
        strategy_maxdd = self.calculate_max_drawdown(self.strategy)

        benchmark_dd_vol = self.calculate_drawdown_volatility(self.benchmark)
        strategy_dd_vol = self.calculate_drawdown_volatility(self.strategy)

        self.end_date = date.fromisoformat(self.benchmark.index[-1])

        benchmark_metrics = {'CAGR%': benchmark_CAGR, 'Vol%': benchmark_Volatility, 'Sharpe': benchmark_SR,
                            'MaxDD%': benchmark_maxdd, 'Start Date': self.start_date, 'End Date': self.end_date,
                            'DD_vol': benchmark_dd_vol}
        strategy_metrics = {'CAGR%': strategy_CAGR, 'Vol%': strategy_Volatility, 'Sharpe': strategy_SR,
                            'MaxDD%': strategy_maxdd, 'Start Date': self.start_date, 'End Date': self.end_date,
                            'DD_vol': strategy_dd_vol}

        summarized_perf = pd.DataFrame({'Strategy': strategy_metrics,'Benchmark': benchmark_metrics})
        summarized_perf.index = ['CAGR%', 'Volatility%', 'Sharpe Ratio', 'Max Drawdown%', 'Start Date', 'End Date',
                                'Drawdown Volatility']

        return summarized_perf

    # Method to plot performance
    def plotPerformance(self):
        plt.figure(figsize=(18, 9), dpi=140)
        self.strategy['Close'].plot(label='Strategy')
        self.benchmark['Close'].plot(label='Benchmark')
        plt.title('Equity Curves', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Close Amount')
        plt.legend()
        st.pyplot(plt)

# Streamlit App
def main():
    st.set_page_config(page_title="Stock Analysis App", layout="wide")

    st.title("Strategy Performance Analysis")

    # Inputs from the user
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    strategy_duration = st.number_input("Number of days for strategy duration", min_value=1, value=30)
    initial_equity = st.number_input("Initial Equity", min_value=0, step=1_000, value=1_000_000)

        # Updated list of stock symbols with NSE exchange
    nifty50_stock = [
            "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
            "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS",
            "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
            "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
            "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS",
            "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS",
            "LT.NS", "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS", "ONGC.NS",
            "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS",
            "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS",
            "TITAN.NS", "UPL.NS", "ULTRACEMCO.NS", "WIPRO.NS"
        ]

    if st.button("Calculate"):
            performance = Portfolio(start_date, end_date, initial_equity, strategy_duration, nifty50_stock)


            # Display selected stocks
            st.subheader("Stocks Selected for Strategy")
            st.write(performance.selected_stocks)

            # Display performance metrics
            st.subheader("Performance Metrics")
            st.write(performance.summarized_perf)

            # Explanation of performance metrics
            st.write(
            "- **CAGR (Compound Annual Growth Rate):** Measures the mean annual growth rate of an investment over a specified time period.\n"
            "- **Volatility:** Indicates the degree of variation in a trading price series over time.\n"
            "- **Sharpe Ratio:** Measures the risk-adjusted performance of an investment.\n"
            "- **Max Drawdown:** The maximum observed loss from a peak to a trough during a specific period.\n"
            "- **Drawdown Volatility:** The standard deviation of drawdowns."
        )
            # Display equity curves
            st.subheader("Equity Curves")
            performance.plotPerformance()

if __name__ == "__main__":
    main()
