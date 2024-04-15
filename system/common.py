import pickle
from slugify import slugify


### model selection & visulisation related
def get_forecast_plot_data(series_df, forecast_df):
    # Plot series history
    line_history = dict(
        type="scatter",
        x=series_df.index,
        y=series_df["value"],
        name="Historical",
        mode="lines+markers",
        line=dict(color="rgba(255, 255, 255,0.8)"),
    )

    forecast_error_x = list(forecast_df.index) + list(
        reversed(forecast_df.index)
    )
    forecast_error_x = [x.to_pydatetime() for x in forecast_error_x]

    print("x:",forecast_error_x)
    print("err75:",list(forecast_df["UB_75"]) + list(reversed(forecast_df["LB_75"])))

    # Plot 50% Error Rate
    error_50 = dict(
        type="scatter",
        x=forecast_error_x,
        y=list(forecast_df["UB_50"]) + list(reversed(forecast_df["LB_50"])),
        fill="tozeroy",
        fillcolor="rgb(226, 87, 78)",
        line=dict(color="rgba(255,255,255,0)"),
        name="50% ER",
    )

    # Plot 75% Error Rate
    error_75 = dict(
        type="scatter",
        x=forecast_error_x,
        y=list(forecast_df["UB_75"]) + list(reversed(forecast_df["LB_75"])),
        fill="tozeroy",
        fillcolor="rgb(234, 130, 112)",
        line=dict(color="rgba(255,255,255,0)"),
        name="75% ER",
    )

    # Plot 95% Error Rate
    error_95 = dict(
        type="scatter",
        x=forecast_error_x,
        y=list(forecast_df["UB_95"]) + list(reversed(forecast_df["LB_95"])),
        fill="tozeroy",
        fillcolor="rgb(243, 179, 160)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% ER",
    )

    # Plot forecast
    line_forecast = dict(
        type="scatter",
        x=forecast_df.index,
        y=forecast_df["forecast"],
        name="Forecast",
        mode="lines",
        line=dict(color="rgba(255,255,255,1)", dash="2px"),
    )

    data = [error_95, error_75, error_50, line_forecast, line_history]

    return data


def get_plot_shapes(series_df, forecast_df):
    shapes = [
        {
            "type": "rect",
            # x-reference is assigned to the x-values
            "xref": "x",
            "x0": series_df.index[0],
            "x1": series_df.index[-1],
            # y-reference is assigned to the plot paper [0,1]
            "yref": "paper",
            "y0": 0,
            "y1": 1,
            "fillcolor": "rgba(206, 212, 220,0.1)",
            # "fillcolor": "rgb(229, 236, 245)",
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "rect",
            # x-reference is assigned to the x-values
            "xref": "x",
            "x0": forecast_df.index[0],
            "x1": forecast_df.index[-1],
            # y-reference is assigned to the plot paper [0,1]
            "yref": "paper",
            "y0": 0,
            "y1": 1,
            "fillcolor": "rgba(206, 212, 220,0)",
            "line": {"width": 0},
            "layer": "below",
        },
    ]

    return shapes


def get_series_figure(data_dict, model_name):

    series_df = data_dict["downloaded_dict"]["series_df"]
    forecast_df = data_dict["all_forecasts"][model_name]["forecast_df"]

    data = get_forecast_plot_data(series_df, forecast_df)
    shapes = get_plot_shapes(series_df, forecast_df)

    time_difference_forecast_to_start = (
        forecast_df.index[-1].to_pydatetime()
        - series_df.index[0].to_pydatetime()
    )

    # title = (
    #     data_dict["data_source_dict"]["short_title"]
    #     if "short_title" in data_dict["data_source_dict"]
    #     else data_dict["data_source_dict"]["title"]
    # )

    layout = dict(
        # title=title,
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,0,0,0)",
        font={"color": "darkgray"},
        xaxis=dict(
            showline=False,
            showgrid=False,
            zeroline=False,
            fixedrange=True,
            type="date",
            range=[
                series_df.index[
                    -16
                ].to_pydatetime(),  # Recent point in history
                forecast_df.index[-1].to_pydatetime(),  # End of forecast range
            ],
            rangeselector=dict(
                buttons=list(
                    [
                        dict(
                            count=5,
                            label="50%",
                            step="year",
                            stepmode="backward",


                        ),
                        dict(
                            count=10,
                            label="80%",
                            step="year",
                            stepmode="backward",
                        ),
                        dict(
                            count=time_difference_forecast_to_start.days,
                            label="all",
                            step="day",
                            stepmode="backward",
                        ),
                    ]
                )
            ),
            rangeslider=dict(
                visible=True,
                range=[
                    series_df.index[0].to_pydatetime(),
                    forecast_df.index[-1].to_pydatetime(),
                ],
            ),
        ),
        yaxis=dict(
            # Will disable all zooming and movement controls if True
            fixedrange=True,
            autorange=True,
            showgrid=False,
            showline=False,
            zeroline=False,
        ),
        shapes=shapes,
        # modebar={"color": "rgba(255,255,255,1)"},
    )

    return dict(data=data, layout=layout)


def get_forecast_data(title):
    title = slugify(title)
    f = open(f"./data/forecasts/{title}.pkl", "rb")
    data_dict = pickle.load(f)
    return data_dict

#test
data_dict=get_forecast_data("gdp-for-andorra-world-bank")
get_series_figure(data_dict,"RNN")