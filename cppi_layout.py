import streamlit as st ##streamplit package don't work with eikon package!!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.stats
from scipy.stats import norm

def main():

    st.title(' CPPI Workbook App')
    st.markdown(
        'Constant Proportion Portfolio Investment (CPPI) is a trading strategy that provides an upside potential of a \
        risky asset while providing a capital guarantee against downside risk.')

    st.sidebar.title("CPPI Model Variables")


    @st.cache(persist=True)
    def load_data():
        ric_history = pd.read_excel('bist_indices_data.xlsx',  index_col='Date', parse_dates=True)
        ric_history_pct = ric_history.pct_change().dropna()
        return ric_history, ric_history_pct

    index_names = pd.DataFrame({'Index Names': {'.XU100': 'BIST 100', '.XU030': 'BIST 30', '.XTUMY': 'BIST ALL -100',
                                                '.XBANK': 'BIST BANKS', '.XUSIN': 'BIST INDUSTRIALS','.XHOLD': 'BIST HOLDING AND INVESTMENT',
                                                '.XBLSM': 'BIST INFO. TECHNOLOGY', '.XULAS': 'BIST TRANSPORTATION', '.XELKT': 'BIST ELECTRICITY',
                                                '.XGIDA': 'BIST FOOD BEVERAGE', '.XTRZM': 'BIST TOURISM'}})

    df_abs, df_pct = load_data()

    def cppi_func(risky_r, riskfree_rate=0.1, m=3, start=1000, floor=0.8, drawdown=None, periods_per_year=52):
        # set up the CPPI parameters
        dates = risky_r.index
        n_steps = len(dates)
        account_value = start
        floor_value = start * floor
        peak = account_value

        if isinstance(risky_r, pd.Series):
            risky_r = pd.DataFrame(risky_r, columns=["R"])

        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r[:] = riskfree_rate / periods_per_year  # fast way to set all values to a number

        # set up some DataFrames for saving intermediate values
        account_history = pd.DataFrame().reindex_like(risky_r)
        risky_w_history = pd.DataFrame().reindex_like(risky_r)
        cushion_history = pd.DataFrame().reindex_like(risky_r)
        floorval_history = pd.DataFrame().reindex_like(risky_r)
        peak_history = pd.DataFrame().reindex_like(risky_r)

        for step in range(n_steps):
            if drawdown is not None:
                peak = np.maximum(peak, account_value)
                floor_value = peak * (1 - drawdown)
            cushion = (account_value - floor_value) / account_value
            risky_w = m * cushion
            risky_w = np.minimum(risky_w, 1)
            risky_w = np.maximum(risky_w, 0)
            safe_w = 1 - risky_w
            risky_alloc = account_value * risky_w
            safe_alloc = account_value * safe_w
            # recompute the new account value at the end of this step

            account_value = risky_alloc * (1 + risky_r.iloc[step]) + safe_alloc * (1 + safe_r.iloc[step])
            # save the histories for analysis and plotting
            cushion_history.iloc[step] = cushion
            risky_w_history.iloc[step] = risky_w
            account_history.iloc[step] = account_value
            floorval_history.iloc[step] = floor_value
            peak_history.iloc[step] = peak

        risky_wealth = start * (1 + risky_r).cumprod()

        backtest_result = {
            "Wealth": account_history,
            "Risky Wealth": risky_wealth,
            "Risk Budget": cushion_history,
            "Risky Allocation": risky_w_history,
            "m": m,
            "start": start,
            "risky_r": risky_r,
            "safe_r": safe_r,
            "drawdown": drawdown,
            "peak": peak_history,
            "floor": floorval_history
        }

        return backtest_result

    def plot_metrics(inp, classifier):
            fig, ax = plt.subplots()
            ax = inp['Wealth'].iloc[:, 0].plot(figsize=(18, 10), color = 'blue', linewidth= 3, label='CPPI Strategy')
            inp['Risky Wealth'].iloc[:, 0].plot(ax=ax, style='k',linewidth= 2, label='If 100% invested in {}'.format(classifier))
            inp['floor'].iloc[:, 0].plot(ax=ax, color='r', linestyle='--', label='Floor Value of the Investment', linewidth= 3)
            ax.set_xlabel('Date', fontsize=20, fontweight = 'bold')
            ax.set_ylabel('Index Values', fontsize=20, fontweight = 'bold')

            ax.tick_params(labelsize=18)
            ax.legend(fontsize="xx-large", frameon = False)
            st.pyplot(fig)

            ## Understanding the distribution
    def annualize_rets(r, periods_per_year=52):
        compounded_growth = (1 + r).prod()
        n_periods = r.shape[0]
        return compounded_growth ** (periods_per_year / n_periods) - 1

    def annualize_vol(r, periods_per_year=52):
        return r.std() * np.sqrt(periods_per_year)

    def sharpe_ratio(r, riskfree=0.0, periods_per_year=52):
        rf_per_period = (1 + riskfree) ** (1 / periods_per_year) - 1

        excess_ret = r - rf_per_period
        ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
        ann_vol = annualize_vol(r, periods_per_year)
        return ann_ex_ret / ann_vol

    def drawdown(return_series: pd.Series):
        wealth_index = 1000 * (1 + return_series).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return pd.DataFrame({"Wealth": wealth_index,
                             "Previous Peak": previous_peaks,
                             "Drawdown": drawdowns})

    def var_historic(r, level=5):
        if isinstance(r, pd.DataFrame):
            return r.aggregate(var_historic, level=level)
        elif isinstance(r, pd.Series):
            return -np.percentile(r, level)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")

    def cvar_historic(r, level=5):
        if isinstance(r, pd.Series):
            is_beyond = (r <= -var_historic(r, level=level))
            return -r[is_beyond].mean()
        elif isinstance(r, pd.DataFrame):
            return r.aggregate(cvar_historic, level=level)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")

    def var_gaussian(r, level=5, modified=False):
        z = norm.ppf(level / 100)
        if modified:  # Cornish-Fisher VaR
            # modify the Z score based on observed skewness and kurtosis
            s = scipy.stats.skew(r)
            k = scipy.stats.kurtosis(r, fisher=False)
            z = (z +
                 (z ** 2 - 1) * s / 6 +
                 (z ** 3 - 3 * z) * (k - 3) / 24 -
                 (2 * z ** 3 - 5 * z) * (s ** 2) / 36
                 )
        return -(r.mean() + z * r.std(ddof=0))

    def summary_stats(r, riskfree=0.00, periods_per_year=52):
        """
        Assumes periods per year is 52 when assuming the data is weekly. If not, change periods_per_year!
        """
        ann_r = r.aggregate(annualize_rets, periods_per_year=periods_per_year)
        ann_vol = r.aggregate(annualize_vol, periods_per_year=periods_per_year)
        ann_sr = r.aggregate(sharpe_ratio, riskfree=0, periods_per_year=periods_per_year)
        dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
        cf_var5 = r.aggregate(var_gaussian, level=5, modified=True)
        hist_cvar5 = r.aggregate(cvar_historic, level=5)
        return pd.DataFrame({
            "Annualized Return %": ann_r*100,
            "Annualized Vol %": ann_vol*100,
            "Cornish-Fisher VaR (5%) %": cf_var5 * 100,
            "Historic CVaR (5%) %": hist_cvar5*100,
            "Sharpe Ratio": ann_sr,
            "Max Drawdown %": dd*100
        })

    #Selecting the CPPI Method
    st.sidebar.subheader("Pick a Method")
    method_name = st.sidebar.selectbox("Methods:", ("Nonparametric CPPI Strategy", "Parametric CPPI Strategy", "G.Brownian Motion CPPI Strategy"))

    if method_name == 'Nonparametric CPPI Strategy':
        # st.sidebar.subheader("Choose Equity Index")
        classifier = st.selectbox("Choose Index Name", list(index_names['Index Names']))
        date_interval = st.date_input("(Optional) Choose Start and End Dates", [df_abs.index.min(), df_abs.index.max()])

        start_date = date_interval[0]
        end_date = date_interval[1]

        clas2 = index_names[index_names['Index Names'] == classifier].index[0]
        ind = df_pct[[clas2]].loc[start_date: end_date]  #Getting the date based on the index name and the date interval

        st.subheader("**Risk Measures Table**")
        st.markdown('*Assumptions*: Risk-free rate=10%, m=3, \n floor=0.8, no drawdown')

        Q = cppi_func(ind)

        # Comparing the risk profile and distribution characteristics
        metric1 = Q['Risky Wealth'].pct_change().dropna()   # Risky Portfolio
        metric2 = Q['Wealth'].pct_change().dropna()  # CPPI Portfolio
        metrics_combined = pd.concat([metric1, metric2], axis=1)
        metrics_combined.columns = ['Risky Portfolio', 'CPPI Portfolio']
        st.write(summary_stats(metrics_combined, riskfree=0.0, periods_per_year=52).T)  ###riskfree 0% !!!!!

        # if st.sidebar.checkbox('Show CPPI plot', False):
        st.subheader("**Nonparametric CPPI Strategy Chart**")
        plot_metrics(Q, classifier=classifier)

    elif method_name == 'Parametric CPPI Strategy':
        classifier = st.selectbox("Choose Index Name", list(index_names['Index Names']))
        date_interval = st.date_input("(Optional) Choose Start and End Dates", [df_abs.index.min(), df_abs.index.max()])

        start_date = date_interval[0]
        end_date = date_interval[1]

        clas2 = index_names[index_names['Index Names'] == classifier].index[0]
        ind = df_pct[[clas2]].loc[start_date: end_date]

        st.sidebar.subheader("Parametric CPPI Strategy\n *Model Hyperparameters*")

        m_ratio = st.sidebar.slider("m (Leverage ratio)", 0.0, 10.0, step=0.5, value= 5.0, key='m_ratio')
        rf_ratio = st.sidebar.slider("Riskfree rate (annual) %", 0.0, 30.0, step=1.0, value= 10.0, key='rf_ratio')
        drawdown_ = st.sidebar.radio("Drawdown from the peak ?", ('Yes', 'No'), key='drawdown_')

        floor_ = 0.80  #needed to prvent error message
        if drawdown_ == 'No':
            drawdown_ratio = None
            floor_ = st.sidebar.slider("Floor %", 0.0, 100.0, step=1.0, value= 80.0, key='floor_')
        else:
            drawdown_ratio = st.sidebar.slider("Drawdown ratio (drawdown from the peak) %", 0.0, 50.0, step=1.0, value= 10.0, key='drawdown')
            drawdown_ratio = drawdown_ratio/100

        R = cppi_func(ind, riskfree_rate=rf_ratio/100, m=m_ratio, start=1000, floor=floor_/100 ,drawdown=drawdown_ratio, periods_per_year=52)

        st.subheader("**Risk Measures Table**")
        # Comparing the risk profile and distribution characteristics
        metric1 = R['Risky Wealth'].pct_change().dropna()   # Risky Portfolio
        metric2 = R['Wealth'].pct_change().dropna()  # CPPI Portfolio
        metrics_combined = pd.concat([metric1, metric2], axis=1)
        metrics_combined.columns = ['Risky Portfolio', 'CPPI Portfolio']
        st.write(summary_stats(metrics_combined, riskfree=0.0, periods_per_year=52).T)

        st.subheader("**Parametric CPPI Strategy Chart**")
        plot_metrics(R, classifier=classifier)

    elif method_name == 'G.Brownian Motion CPPI Strategy':
        st.sidebar.subheader("GBM CPPI Strategy\n *Model Hyperparameters*")

        n_scenarios = st.sidebar.slider("Number of scenarios", 1000, 50000, step=1000, value=10000, key='n_scenarios')
        n_years = st.sidebar.slider("Number of years", 1, 10, step=1, value=3, key='n_years')
        mu = st.sidebar.slider("Expected annual return", 0.0, 0.3, step=0.1, value=0.1, key='mu')
        sigma = st.sidebar.slider("Expected annual volatility", 0.0, 0.5, step=0.5, value=0.25, key='sigma')

        m_ratio = st.sidebar.slider("m (Leverage ratio)", 0.0, 10.0, step=0.5, value=3.0, key='m_ratio')
        rf_ratio = st.sidebar.slider("Riskfree rate (annual) %", 0.0, 30.0, step=1.0, value=10.0, key='rf_ratio')
        rebalance = st.sidebar.number_input("Rebalance/year", 1, 252, step=1, value=12, key='rebals')
        drawdown_ = st.sidebar.radio("Drawdown from the peak ?", ('Yes', 'No'), key='drawdown_')

        floor_ = 0.80  # needed to prevent error message
        if drawdown_ == 'No':
            drawdown_ratio = None
            floor_ = st.sidebar.slider("Floor %", 0.0, 100.0, step=1.0, value=80.0, key='floor_')
        else:
            drawdown_ratio = st.sidebar.slider("Drawdown ratio (drawdown from the peak) %", 0.0, 50.0, step=1.0,
                                               value=10.0, key='drawdown')
            drawdown_ratio = drawdown_ratio / 100

        #creating random numbers
        def geometric_brownian_motion(n_scenarios=n_scenarios, steps_per_year = rebalance, n_years=n_years, s0=100,
                                        mu=mu, sigma=sigma, prices=True):
            n_steps = int(n_years * steps_per_year) + 1
            dt = 1 / steps_per_year

            rets_plus_1 = np.random.normal(loc=1 + mu * dt, scale=sigma * np.sqrt(dt), size=(n_steps, n_scenarios))
            rets_plus_1[0] = 1

            ret_val = s0 * pd.DataFrame(rets_plus_1).cumprod() if prices else pd.DataFrame(rets_plus_1) - 1

            return ret_val

        rets = geometric_brownian_motion(prices=False)

        G = cppi_func(rets, riskfree_rate=rf_ratio / 100, m=m_ratio, start=1000, drawdown=drawdown_ratio,
                      floor=floor_ / 100, periods_per_year=12)

        st.subheader("**Geometric Brownian Motion CPPI Strategy Chart**")
        plot_metrics(G, classifier='Brownian Risky Asset')


    st.sidebar.subheader("> Data Source")
    if st.sidebar.checkbox('Show raw data', False):
        st.sidebar.markdown('*Given at the end of the page!*')
        st.subheader('Equity Indices Adjusted Closing Data')
        st.write(df_abs[::-1])
        st.write(index_names.T)


    st.markdown('......................................................................................................\
    .........................................................')
    st.markdown("*CPPI strategy:* At every point in time, take a look at *cushion* which is the difference between the asset\
     value and a given *floor* ( minimum desired level for the assets.) Then allocate to the risky asset \
    by a *multiple M* of that cushion.  When the cash goes down, reduce the risky allocation." )

    st.markdown(
        "Diversification cannot help manage Systematic Risk. Hedging is effective at managing Systematic Risk, but \
        gives up a lot of uspide. CPPI is the best of all!")
    st.markdown('.....................................................................................................\
    .........................................................')

    st.sidebar.markdown('<a href="mailto:eg621@yahoo.com">Contact me !</a>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()



