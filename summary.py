import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.stattools as ts
import warnings


def process_data(weather_data_filepath, electricity_data_filepath):
    df_weather = pd.read_csv(weather_data_filepath, delimiter=",")
    df_electricity = pd.read_csv(
        electricity_data_filepath, delimiter=","
    )
    df_weather["time"] = pd.to_datetime(df_weather["referenceTime"])
    df_electricity["time"] = pd.to_datetime(
        df_electricity["måned"], format="%YM%m"
    )
    df_weather = df_weather.drop(columns="referenceTime")
    df_electricity = df_electricity.drop(columns="måned")
    df_weather["month_year"] = df_weather["time"].dt.strftime("%Y-%m")
    df_electricity["month_year"] = df_electricity["time"].dt.strftime(
        "%Y-%m"
    )
    df_weather["rolling_mean_precipitation"] = (
        df_weather["value"].rolling(window=6).mean()
    )
    df_electricity["rolling_mean_electricity"] = (
        df_electricity["value"].rolling(window=6).mean()
    )
    df_filtered_weather = df_weather[
        (
            df_weather["month_year"]
            >= df_electricity["month_year"].min()
        )
        & (
            df_weather["month_year"]
            <= df_electricity["month_year"].max()
        )
    ]
    df_filtered_weather.reset_index(drop=True, inplace=True)

    df_merged = pd.merge(
        df_weather.drop(columns="time"),
        df_electricity.drop(columns="time"),
        on="month_year",
        suffixes=["_precipitation", "_electricity"],
        how="outer",
    )
    df_merged.sort_values(by="month_year", inplace=True)

    return df_weather, df_electricity, df_merged


def get_cropped_df(df, start_year, end_year):
    df["month_year"] = pd.to_datetime(
        df["month_year"], format="%Y-%m"
    )
    cropped_df_merged = df[
        (df["month_year"].dt.year >= start_year)
        & (df["month_year"].dt.year <= end_year)
    ]
    cropped_df_merged.reset_index(drop=True, inplace=True)
    return cropped_df_merged


def plot_values(
    data: pd.DataFrame,
    x_var="time",
    y_var="value",
    title: str = None,
    x_label="Time",
    y_label="Value",
):
    plt.figure(figsize=(12, 6))
    plt.plot(data[x_var], data[y_var], marker="o")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()


def print_data_summary(df: pd.DataFrame):
    summary = df["value"].describe()
    print("\n")
    print(f"Data Summary:")
    print(summary)
    print("\n")


def plot_rolling_mean_together(
    df: pd.DataFrame, start_year: int, end_year: int
):
    df = get_cropped_df(df, start_year, end_year)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(
        df["month_year"],
        df["rolling_mean_precipitation"],
        color="blue",
        label="Precipitation",
    )
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Precipitation (mm)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=12))

    ax2 = ax1.twinx()
    ax2.plot(
        df["month_year"],
        df["rolling_mean_electricity"],
        color="green",
        label="Electricity",
    )
    ax2.set_ylabel("Electricity production(MWh)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    if end_year - start_year == 0:
        plt.title(
            f"""Precipitation and electricity production 
            6-months rolling windows for year {start_year}"""
        )
    else:
        plt.title(
            f"""Precipitation and electricity production
             6-months rolling windows for years {start_year, end_year}"""
        )
    fig.tight_layout()
    fig.legend()
    plt.show()


def print_adf(df: pd.DataFrame):
    adf_result = ts.adfuller(df["value"])
    print("ADF Statistic:", adf_result[0])
    print("p-value:", adf_result[1])
    print("Critical Values:")
    for key, value in adf_result[4].items():
        print(f"   {key}: {value}")
    print("\n")


def plot_monthly_means(df: pd.DataFrame, ylabel: str, title: str):
    df = df.reset_index(drop=False)
    df = df.set_index("month_year", drop=False)
    df.index = pd.to_datetime(df.index, format="%Y-%m")
    monthly_avg = df.groupby(df.index.month)["value"].mean()
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 13), monthly_avg, marker="o")
    plt.xticks(
        range(1, 13),
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )
    plt.xlabel("Month")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_acf(df: pd.DataFrame, xlim: tuple, ylim: tuple):
    plt.figure(figsize=(12, 6))
    autocorrelation_plot(df["value"])
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


def plot_decomposed(df: pd.DataFrame, period: int):
    decomp = seasonal_decompose(
        df["value"], model="additive", period=period
    )
    decomp.plot()
    plt.show()


def plot_diffed_series(df: pd.DataFrame, lag: int):
    df["value_diff"] = df["value"].diff(lag)
    df.dropna()
    plt.figure(figsize=(12, 6))
    plt.plot(df["value_diff"])
    plt.show()


def evaluate_arima_model(X, order=(2, 0, 1)):
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    error = mean_squared_error(test, predictions)
    return error


def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype("float32")
    sorted_models = {}
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                        print("ARIMA%s MSE=%.3f" % (order, mse))
                    sorted_models[f"ARIMA{order}"] = mse
                except:
                    continue
    sorted_models = {
        k: v
        for k, v in sorted(
            sorted_models.items(), key=lambda item: item[1]
        )
    }
    print("Sorted from best to worst:")
    for model, mse in sorted_models.items():
        print(f"{model}: MSE={mse:.3f}")
    print("\n")
    print("Best ARIMA%s MSE=%.3f" % (best_cfg, best_score))


def plot_insample_preds(
    preds: pd.DataFrame, ground_truth: pd.DataFrame
):
    plt.figure(figsize=(12, 6))
    plt.plot(ground_truth["value"], label="Ground truth")
    plt.plot(preds, label="In-Sample Predictions", color="red")
    plt.xlabel("Date")
    plt.ylabel("Electricity Production (mWh)")
    plt.title("ARMA Model In-Sample Predictions")
    plt.legend()
    plt.show()


def plot_forecast(model_fit, steps: int, ground_truth: pd.DataFrame):
    forecast = model_fit.forecast(steps=steps)
    plt.figure(figsize=(12, 6))
    plt.plot(
        range(0, len(ground_truth)),
        ground_truth["value"],
        label="Ground truth",
    )
    plt.plot(
        range(len(ground_truth), len(ground_truth) + steps),
        forecast,
        color="green",
        label="forecast",
    )
    plt.legend()
    plt.show()


def plot_forecast_with_exog(
    forecast: pd.DataFrame, exog: pd.DataFrame
):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("Month")
    ax1.plot(forecast, label="Electricity forecast", color="red")
    ax1.set_ylabel(
        "ARMAX forecasted electricity production", color="red"
    )
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.grid(color="pink")
    exog.index = forecast.index
    ax2 = ax1.twinx()
    ax2.plot(exog, label="Precipitation", color="blue")
    ax2.set_ylabel("Forecasted precipitation", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    fig.tight_layout()
    fig.legend()
    plt.show()


def main():
    WEATHER_DATA_FILEPATH = "data/weather_data.csv"
    ELECTRICITY_DATA_FILEPATH = "data/electricity_production_data.csv"
    df_weather, df_electricity, df_merged = process_data(
        WEATHER_DATA_FILEPATH, ELECTRICITY_DATA_FILEPATH
    )
    df_merged_all_values = df_merged.dropna()

    # PART 1: Exploration
    pre_millenium_cropped_weather_data = get_cropped_df(
        df_weather, 1998, 1999
    )
    pre_millenium_cropped_el_data = get_cropped_df(
        df_electricity, 1998, 1999
    )
    plot_values(
        pre_millenium_cropped_weather_data,
        title="Weather data",
        y_label="Precipitation(mm)",
    )
    plot_values(
        pre_millenium_cropped_el_data,
        title="Electricity production",
        y_label="Electricity(MWh)",
    )
    plot_rolling_mean_together(df_merged, 2006, 2010)
    print("Precipitation")
    print_data_summary(df_weather)
    print("Electricity production")
    print_data_summary(df_electricity)
    correlation = df_merged["value_precipitation"].corr(
        df_merged["value_electricity"]
    )
    print(
        f"""\nCorrelation between Precipitation 
        and Electricity Production: {correlation}"""
    )

    # PART 2: Transformations
    print("Electricity ADF:")
    print_adf(df_electricity)
    print("Weather ADF:")
    print_adf(df_weather)
    plot_monthly_means(
        df_electricity,
        "Average electricity production (MWh)",
        "Average monthly Electricity production",
    )
    plot_monthly_means(
        df_weather,
        "Average precipitation(mm)",
        "Average monthly Precipitation",
    )
    plot_acf(df_electricity, (0, 230), (-0.5, 0.5))
    plot_acf(df_weather, (0, 300), (-0.5, 0.5))
    plot_decomposed(df_electricity, 12)
    plot_decomposed(df_weather, 12)
    plot_diffed_series(df_electricity, 12)
    plot_diffed_series(df_weather, 12)

    # PART 3: Modeling
    AR = 2
    DIFF = 0
    MA = 1
    order = (AR, DIFF, MA)
    params = {"order": order}
    p_values_grid = list(range(0, 3)) + [12]
    d_values_grid = [0, 1, 6, 12]
    q_values_grid = list(range(0, 3)) + [12]
    warnings.filterwarnings("ignore")
    evaluate_models(
        df_electricity["value"].values,
        p_values_grid,
        d_values_grid,
        q_values_grid,
    )
    model = ARIMA(df_electricity["value"], **params)
    model_fit = model.fit()
    print(model_fit.summary())
    in_sample_preds = model_fit.predict(
        start=0, end=len(df_electricity) - 1, dynamic=False
    )
    plot_insample_preds(in_sample_preds, df_electricity)
    plot_forecast(model_fit, 36, df_electricity)

    ## PART 3.2: ARMAX
    exog = get_cropped_df(df_weather, 1993, 2011)["value"]
    params["exog"] = exog
    model_exog = ARIMA(df_electricity["value"], **params)
    warnings.filterwarnings("ignore")
    model_exog_fit = model_exog.fit()
    print(model_exog_fit.summary())
    forecast_weather_data = get_cropped_df(df_weather, 2012, 2013)[
        "value"
    ]
    exog_forecast = model_exog_fit.forecast(
        steps=24, exog=forecast_weather_data
    )
    plot_forecast_with_exog(exog_forecast, forecast_weather_data)
    armax_insample_preds = model_exog_fit.predict(
        start=0, end=len(df_electricity) - 1, dynamic=False
    )
    plot_insample_preds(armax_insample_preds, df_electricity)


main()
