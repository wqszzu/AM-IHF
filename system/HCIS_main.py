import os
import pathlib

import dash
from dash import dcc
from dash import html
# import dash_core_components as dcc
# import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import feffery_antd_components as fac
import random

# import dash_table
import plotly.graph_objs as go
import dash_daq as daq

import pandas as pd

import pickle
from slugify import slugify



import numpy as np
import datetime as dt
from dash.exceptions import PreventUpdate
from scipy.stats import rayleigh
from db.api import get_wind_data, get_wind_data_by_id

import PythonAPI.util.globalvar as gl

GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 5000)


LOWER_SIZE_LIMIT=5
UPPER_SIZE_LIMIT=25
LOWER_WINDOW_SIZE=0
UPPER_WINDOW_SIZE=50

#global data
gl._init()
gl.set_value('GLOBAL_DATA',["None"])

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Human-computer interaction system"
server = app.server
app.config["suppress_callback_exceptions"] = True
#082255
app_color = {"rl_graph_bg": "rgba(0,0,0,0.1)", "rl_graph_line": "#007ACE"}
# rgba(0,0,0,0.3)

APP_PATH = str(pathlib.Path(__file__).parent.resolve())
df = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "spc_data.csv")))

params = list(df)
max_length = len(df)

suffix_row = "_row"
suffix_button_id = "_button"
suffix_sparkline_graph = "_sparkline_graph"
suffix_count = "_count"
suffix_ooc_n = "_OOC_number"
suffix_ooc_g = "_OOC_graph"
suffix_indicator = "_indicator"


def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    fac.AntdHeader(
                        fac.AntdTitle(

                            'HCIS',
                            level=1,
                            style={
                                'fontFamily': "Times New Roman",
                                'color': 'white',
                                "margin-left":"-80px",
                                "margin-top":"20px"
                            }
                        ),
                        style={
                            'display': 'flex',
                            'justifyContent': 'left',
                            'alignItems': 'center',
                            'backgroundColor': 'rgba(0, 0, 0, 0)'


                        },

                    ),
                    # html.H6("Real-time Monitoring and Process Control"),
                ],
            ),
        ],
    )


def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="se_tab",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="simulation-environment-tab",
                        label="Simulation Environment",
                        value="se_tab",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="rl-decision-tab",
                        label="RL Decision",
                        value="rl_tab",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),

                ],
            )
        ],
    )


def init_df():
    ret = {}
    for col in list(df[1:]):
        data = df[col]
        stats = data.describe()

        std = stats["std"].tolist()
        ucl = (stats["mean"] + 3 * stats["std"]).tolist()
        lcl = (stats["mean"] - 3 * stats["std"]).tolist()
        usl = (stats["mean"] + stats["std"]).tolist()
        lsl = (stats["mean"] - stats["std"]).tolist()

        ret.update(
            {
                col: {
                    "count": stats["count"].tolist(),
                    "data": data,
                    "mean": stats["mean"].tolist(),
                    "std": std,
                    "ucl": round(ucl, 3),
                    "lcl": round(lcl, 3),
                    "usl": round(usl, 3),
                    "lsl": round(lsl, 3),
                    "min": stats["min"].tolist(),
                    "max": stats["max"].tolist(),
                    "ooc": populate_ooc(data, ucl, lcl),
                }
            }
        )

    return ret


def populate_ooc(data, ucl, lcl):
    ooc_count = 0
    ret = []
    for i in range(len(data)):
        if data[i] >= ucl or data[i] <= lcl:
            ooc_count += 1
            ret.append(ooc_count / (i + 1))
        else:
            ret.append(ooc_count / (i + 1))
    return ret


state_dict = init_df()


def init_value_setter_store():
    # Initialize store data
    state_dict = init_df()
    return state_dict

#build simulation enviroment layout
def build_se_tab(stopped_interval):
    return [
        html.Div(
            id="status-container",
            children=[
                build_quick_stats_panel(),
                html.Div(
                    id="graphs-container",
                    children=[build_vehicle_condition_panel(stopped_interval),
                              build_weather_condition_panel(),
                              build_task_forcast_panel(),
                              ],
                ),
            ],
        ),
    ]

def get_current_time():
    """ Helper function to get the current time in seconds. """

    now = dt.datetime.now()
    total_time = (now.hour * 3600) + (now.minute * 60) + (now.second)
    return total_time

#build rl decision layout
def build_rl_tab():
    return [
        # Manually select metrics
        html.Div(
            id="set-specs-intro-container",
            # className='twelve columns',
            # children=html.P(
            #     # "Real-time monitoring of RL training, and adjusting the time window range to improve learning quality."
            # ),
        ),

        #time windows
        html.Div(
            [
                # wind speed
                html.Div(
                    [
                        html.Div(
                            [html.H6("Time Windows", style={"color":"#FFF","font-size":"1.8rem","margin-left":"15px"})]
                        ),
                        dcc.Graph(
                            id="time-window",
                            figure=dict(
                                layout=dict(
                                    plot_bgcolor=app_color["rl_graph_bg"],
                                    paper_bgcolor=app_color["rl_graph_bg"],
                                )
                            ),
                        ),
                        dcc.Interval(
                            id="time-window-update",
                            interval=int(GRAPH_INTERVAL),
                            n_intervals=0,
                        ),
                    ],
                    className="two-thirds column wind__speed__container",
                ),
                html.Div(
                    [
                        # histogram
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "Train Condition",
                                            style={"color": "#FFF", "font-size": "1.8rem", "margin-left": "0px"},
                                        ),

                                    ]
                                ),
                                html.Div(
                                    [
                                        dcc.Slider(
                                            id="bin-slider",
                                            min=1,
                                            max=60,
                                            step=1,
                                            value=20,
                                            updatemode="drag",
                                            marks={
                                                20: {"label": "20"},
                                                40: {"label": "40"},
                                                60: {"label": "60"},
                                            },
                                        )
                                    ],
                                    style={"display": "none"},
                                    className="slider",
                                ),
                                html.Div(
                                    [
                                        dcc.Checklist(
                                            id="bin-auto",
                                            options=[
                                                {"label": "Auto", "value": "Auto"}
                                            ],
                                            value=["Auto"],
                                            inputClassName="auto__checkbox",
                                            labelClassName="auto__label",
                                        ),
                                        html.P(
                                            "# of Bins: Auto",
                                            id="bin-size",
                                            className="auto__p",
                                        ),
                                    ],
                                    style={"display": "none"},
                                    className="auto__container",
                                ),
                                fac.AntdSpace(
                                    [
                                        fac.AntdText('step:', keyboard=True,style={"color": "#FFF"}),
                                        fac.AntdProgress(
                                            id="train-step-percent",
                                            percent=0,
                                            strokeColor={
                                                'from': '#81ffef',
                                                'to': ' #f067b4'
                                            },
                                            showInfo=False,
                                            trailColor='rgba(255,255,255,0.7)',
                                            style={
                                                'width': 700,
                                            }
                                        ),
                                        fac.AntdText("0/0",id="train-step-text", code=True,style={"color": "#FFF"}),
                                    ],
                                ),
                                html.Br(),
                                dcc.Graph(
                                    id="window-train",
                                    figure=dict(
                                        layout=dict(
                                            plot_bgcolor=app_color["rl_graph_bg"],
                                            paper_bgcolor=app_color["rl_graph_bg"],
                                        )
                                    ),
                                ),
                            ],
                            className="graph__container first",
                        ),
                        # window size
                        html.Div(
                            [
                                html.H6(
                                    "Window Size Adjustment",  style={"color":"#FFF","font-size":"1.8rem","margin-left":"0px","margin-top":"20px"}
                                ),

                                html.Div(
                                    id="settings-menu",
                                    children=[

                                        html.Div(
                                            id="value-setter-menu",

                                            children=[
                                                html.Div(id="value-setter-panel",
                                                children=[
                                                    html.Div(
                                                        id="value-setter-panel-header",
                                                        children=[
                                                            html.Label("Specs", className="four columns"),
                                                            html.Label("Current Value", className="four columns"),
                                                            html.Div("Set new value", className="four columns"),
                                                        ],
                                                        className="row",
                                                    ),
                                                    html.Div(
                                                        id="value-setter-panel-usl",
                                                        children=[
                                                            html.Label("Upper Size limit", className="four columns"),
                                                            html.Label(UPPER_SIZE_LIMIT, id="usl-value",
                                                                       className="four columns"),
                                                            html.Div(ud_usl_input, className="four columns"),
                                                        ],
                                                        className="row",
                                                    ),
                                                    html.Div(
                                                        id="value-setter-panel-lsl",
                                                        children=[
                                                            html.Label("Lower Size limit", className="four columns"),
                                                            html.Label(LOWER_SIZE_LIMIT, id="lsl-value",
                                                                       className="four columns"),
                                                            html.Div(ud_lsl_input, className="four columns"),
                                                        ],
                                                        className="row",
                                                    ),]
                                                ),
                                                html.Br(),
                                                html.Div(
                                                    id="button-div",
                                                    children=[
                                                        html.Button("Update", id="value-setter-update-btn"),
                                                        html.Button(
                                                            "Reset",
                                                            id="value-setter-reset-btn",
                                                            n_clicks=0,
                                                        ),
                                                        html.Div(
                                                            id='notification-div'
                                                        )
                                                    ],
                                                ),
                                                # html.Div(
                                                #     id="value-setter-view-output", className="output-datatable"
                                                # ),
                                            ],
                                        ),

                                    ],
                                    style={"margin-top":"-15px"}
                                ),

                            ],
                            className="graph__container second",
                        ),
                    ],
                    className="one-third column histogram__direction",
                ),
            ],
            className="app__content",
        ),

    ]



ud_usl_input = daq.NumericInput(
    id="ud_usl_input", className="setting-input", size=200, max=9999999
)
ud_lsl_input = daq.NumericInput(
    id="ud_lsl_input", className="setting-input", size=200, max=9999999
)


def build_value_setter_line(line_num,value_id, label, value, col3):
    return html.Div(
        id=line_num,
        children=[
            html.Label(label, className="four columns"),
            html.Label(value,id=value_id,className="four columns"),
            html.Div(col3, className="four columns"),
        ],
        className="row",
    )



def build_quick_stats_panel():
    return html.Div(
        id="quick-stats",
        className="row",
        children=[
            html.Div(
                id="card-1",
                children=[
                    html.P("Distribution Overview"),
                    dcc.Graph(id="distribution_overview_graph"),
                    daq.LEDDisplay(
                        id="operator-led",
                        value="1704",
                        color="#92e0d3",
                        backgroundColor="#1e2130",
                        size=50,
                        style={"display": "none"}
                    ),
                ],
            ),
            html.Div(
                id="card-2",
                children=[
                    html.P("Traffic Overview"),
                    dcc.Graph(id="traffic_overview_graph"),
                    daq.Gauge(
                        id="progress-gauge",
                        max=max_length * 2,
                        min=0,
                        showCurrentValue=True,  # default size 200 pixel
                        style={"display": "none"}
                    ),
                ],
            ),
            html.Div(
                id="start-button",
                children=[daq.StopButton(id="stop-button", size=160, n_clicks=0)],
                style={"display":"none"}
            ),
        ],
    )


def generate_section_banner(title):
    return html.Div(className="section-banner", children=title)


def build_vehicle_condition_panel(stopped_interval):
    return html.Div(
        id="vehicle-condition",
        className="row",
        children=[
            # Metrics summary
            html.Div(
                id="metric-summary-session",
                className="eight columns",
                children=[
                    generate_section_banner("Worker Condition"),
                    html.Div(
                        id="metric-div",
                        children=[
                            generate_metric_list_header(),
                            html.Div(
                                id="metric-rows",
                                children=[
                                    generate_metric_row_helper(stopped_interval, 1),
                                    generate_metric_row_helper(stopped_interval, 2),
                                    generate_metric_row_helper(stopped_interval, 3),
                                    generate_metric_row_helper(stopped_interval, 4),
                                    generate_metric_row_helper(stopped_interval, 5),
                                    generate_metric_row_helper(stopped_interval, 6),
                                    generate_metric_row_helper(stopped_interval, 7),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            # Piechart
            html.Div(
                id="vehicle-revenue",
                className="four columns",
                children=[
                    generate_section_banner("Worker Revenue"),
                    generate_piechart(),
                ],


            ),
        ],
    )


def generate_piechart():
    return dcc.Graph(
        id="worker_revenue_piechart",
        figure={
            "data": [
                {
                    "labels": [],
                    "values": [],
                    "type": "pie",
                    "marker": {"line": {"color": "white", "width": 1}},
                    "hoverinfo": "label",
                    "textinfo": "label",
                }
            ],
            "layout": {
                "margin": dict(l=10, r=10, t=10, b=10),
                "showlegend": True,
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
                "font": {"color": "white"},
                "autosize": True,
            },
        },
        config={
            # "modeBarButtonsToRemove": [
            #     "sendDataToCloud",
            #     "autoScale2d",
            #     "hoverClosestCartesian",
            #     "hoverCompareCartesian",
            #     "lasso2d",
            #     "select2d",
            #     "toggleSpikelines",
            # ],
            "displaylogo": False,
        },
        style={
            "width": "85%",
            "margin-top":"30px",
            "margin-left":"50px"
        }
    )


# Build header
def generate_metric_list_header():
    return generate_metric_row(
        "metric_header",
        {"height": "3rem", "margin": "1rem 0", "textAlign": "center"},
        {"id": "m_header_1", "children": html.Div("#")},
        {"id": "m_header_2", "children": html.Div("Name")},
        {"id": "m_header_3", "children": html.Div("Speed")},
        {"id": "m_header_4", "children": html.Div("Task_Num")},
        {"id": "m_header_5", "children": html.Div("Deadline")},
        {"id": "m_header_6", "children": "State"},
    )

def generate_metric_row(id, style, col1, col2, col3, col4, col5, col6):
    if style is None:
        style = {"height": "8rem", "width": "100%"}

    return html.Div(
        id=id,
        className="row metric-row",
        style=style,
        children=[
            html.Div(
                id=col1["id"],
                className="one column",
                style={"textAlign": "center"},
                children=col1["children"],
            ),
            html.Div(
                id=col2["id"],
                style={"textAlign": "center"},
                className="one column",
                children=col2["children"],
            ),
            html.Div(
                id=col3["id"],
                style={"height": "100%","textAlign": "center"},
                className="four columns",
                children=col3["children"],
            ),
            html.Div(
                id=col4["id"],
                style={"textAlign": "center"},
                className="one column",
                children=col4["children"],
            ),
            html.Div(
                id=col5["id"],
                style={"height": "100%", "margin-top": "5rem","textAlign": "center"},
                className="three columns",
                children=col5["children"],
            ),
            html.Div(
                id=col6["id"],
                style={"display": "flex", "justifyContent": "center"},
                className="one column",
                children=col6["children"],
            ),
        ],
    )

def generate_metric_row_helper(stopped_interval, index):
    item = params[index]

    div_id = item + suffix_row
    button_id = item + suffix_button_id
    sparkline_graph_id = item + suffix_sparkline_graph
    count_id = item + suffix_count
    ooc_percentage_id = item + suffix_ooc_n
    ooc_graph_id = item + suffix_ooc_g
    indicator_id = item + suffix_indicator

    return generate_metric_row(
        div_id,
        None,
        {
            "id": item,
            "className": "metric-row-button-text",
            "children": html.Button(
                id=button_id,
                className="metric-row-button",
                children=item,
                title="Click to visualize Weather Condition",
                n_clicks=0,
            ),
        },
        {"id": count_id, "children": "0"},
        {
            "id": item + "_sparkline",
            "children": dcc.Graph(
                id=sparkline_graph_id,
                style={"width": "100%", "height": "95%"},
                config={
                    "staticPlot": False,
                    "editable": False,
                    "displayModeBar": False,
                },
                figure=go.Figure(
                    {
                        "data": [
                            {
                                "x": state_dict["Batch"]["data"].tolist()[
                                    :stopped_interval
                                ],
                                "y": state_dict[item]["data"][:stopped_interval],
                                "mode": "lines+markers",
                                "name": item,
                                "line": {"color": "#f4d44d"},
                            }
                        ],
                        "layout": {
                            "uirevision": True,
                            "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                            "xaxis": dict(
                                showline=False,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                            ),
                            "yaxis": dict(
                                showline=False,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                            ),
                            "paper_bgcolor": "rgba(0,0,0,0)",
                            "plot_bgcolor": "rgba(0,0,0,0)",
                        },
                    }
                ),
            ),
        },
        {"id": ooc_percentage_id, "children": "0.00%"},
        {
            "id": ooc_graph_id + "_container",
            "children": daq.GraduatedBar(
                id=ooc_graph_id,
                color={
                    "ranges": {
                        "#ff3300": [0, 2],
                        "#ffc773 ": [2, 7],
                        "#057748": [7, 10],
                    }
                },
                showCurrentValue=False,
                max=15,
                value=0,
            ),
        },
        {
            "id": item + "_pf",
            "children": daq.Indicator(
                id=indicator_id, value=True, color="#91dfd2", size=12
            ),
        },
    )


def build_weather_condition_panel():
    return html.Div(
        id="weather-condition",
        className="twelve columns",
        children=[
            generate_section_banner("Weather Condition"),
            dcc.Graph(
                id="weather-condition-chart",
                figure=go.Figure(
                    {
                        "data": [
                            {
                                "x": [],
                                "y": [],
                                "mode": "lines+markers",
                                "name": "weather",
                            }
                        ],
                        "layout": {
                            "paper_bgcolor": "rgba(0,0,0,0)",
                            "plot_bgcolor": "rgba(0,0,0,0)",
                            "xaxis": dict(
                                showline=False, showgrid=False, zeroline=False
                            ),
                            "yaxis": dict(
                                showgrid=False, showline=False, zeroline=False
                            ),
                            "autosize": True,
                        },
                    }
                ),
                config={
                    # "modeBarButtonsToRemove": [
                    #     "sendDataToCloud",
                    #     "autoScale2d",
                    #     "hoverClosestCartesian",
                    #     "hoverCompareCartesian",
                    #     "lasso2d",
                    #     "select2d",
                    #     "toggleSpikelines",
                    # ],
                    "displaylogo": False,
                },
            ),
        ],
    )


def build_task_forcast_panel():
    return html.Div(
        id="task-forcast",
        className="twelve columns",
        children=[
            generate_section_banner("Future Task Condition"),
            html.Div(
                [

                    fac.AntdSpace(
                        [
                            fac.AntdText("Forecasting Methods:",style={"color": "darkgray","margin-left":"20px"}),

                            fac.AntdSelect(
                                defaultValue="ARIMA",
                                options=[
                                    {
                                        'label': f'{i}',
                                        'value': f'{i}'
                                    }
                                    for i in ["RNN", "LSTM", "ARIMA", "SVR"]
                                ],
                                style={
                                    'width': 100
                                }
                            ),
                            fac.AntdText("Historical Sequence Length:",style={"color": "darkgray","margin-left":"40px"}),
                            fac.AntdButton(
                                '12',
                                id='button-history-tasks-12',


                            ),
                            fac.AntdButton(
                                '15',
                                id='button-history-tasks-15',
                                # style={"background-color":"#ff5722"}
                                type="primary"
                            ),
                            fac.AntdButton(
                                '24',
                                id='button-history-tasks-24',


                            ),
                            fac.AntdButton(
                                '36',
                                id='button-history-tasks-36',


                            ),
                            fac.AntdButton(
                                '48',
                                id='button-history-tasks-48',


                            ),

                        ],
                        style={

                            "margin-top":"20px"
                        }

                    ),

                    dbc.Col(id="forcast_graph", lg=8,style={"margin-top":"-60px"})
                ]
            )

        ],
    )


# visulisation task forecase graph
def get_forecast_plot_data(history_data, future_data):

    task_num_data=history_data+future_data
    time_slice_list=[]
    for i in range(len(task_num_data)):
        time_slice_list.append(i)


    U_ER_x = [len(history_data)]
    U_ER_25 = [future_data[0]]
    U_ER_50 = [future_data[0]]
    U_ER_75 = [future_data[0]]
    L_ER_x = [len(history_data)]
    L_ER_25 = [future_data[0]]
    L_ER_50 = [future_data[0]]
    L_ER_75 = [future_data[0]]


    for i in range(len(history_data)+1,len(task_num_data)):
        U_ER_x.append(i)

        U_ER_25.append(task_num_data[i] + 1)
        U_ER_50.append(task_num_data[i] + 2)
        U_ER_75.append(task_num_data[i] + 4)

    for i in range(len(history_data)+1, len(task_num_data)):
        L_ER_x.append(i)

        if task_num_data[i]-1<0:
            L_ER_25.append(0)
        else:
            L_ER_25.append(task_num_data[i] - 1)
        if task_num_data[i]-2<0:
            L_ER_50.append(0)
        else:
            L_ER_50.append(task_num_data[i] - 2)
        if task_num_data[i]-4<0:
            L_ER_75.append(0)
        else:
            L_ER_75.append(task_num_data[i] - 4)

    ER_x=list(U_ER_x) + list(reversed(L_ER_x))
    ER_25 = list(U_ER_25) + list(reversed(L_ER_25))
    ER_50 = list(U_ER_50) + list(reversed(L_ER_50))
    ER_75 = list(U_ER_75) + list(reversed(L_ER_75))


    # Plot 25% Error Rate
    error_25 = dict(
        type="scatter",
        x=ER_x,
        y=ER_25,
        fill="tozeroy",
        fillcolor="rgb(226, 87, 78)",
        line=dict(color="rgba(255,255,255,0)"),
        name="25% ER",
    )

    # Plot 50% Error Rate
    error_50 = dict(
        type="scatter",
        x=ER_x,
        y=ER_50,
        fill="tozeroy",
        fillcolor="rgb(234, 130, 112)",
        line=dict(color="rgba(255,255,255,0)"),
        name="50% ER",
    )

    # Plot 75% Error Rate
    error_75 = dict(
        type="scatter",
        x=ER_x,
        y=ER_75,
        fill="tozeroy",
        fillcolor="rgb(243, 179, 160)",
        line=dict(color="rgba(255,255,255,0)"),
        name="75% ER",
    )

    # Plot series history
    line_history = dict(
        type="scatter",
        x=time_slice_list[:len(history_data)+1],
        y=task_num_data[:len(history_data)+1],
        name="Historical",
        mode="lines+markers",
        line=dict(color="rgba(255, 255, 255,0.8)"),
    )

    # Plot forecast
    line_forecast = dict(
        type="scatter",
        x=time_slice_list[len(history_data):],
        y=task_num_data[len(history_data):],
        name="Forecast",
        mode="lines",
        line=dict(color="rgba(255,255,255,1)", dash="2px"),
    )

    data = [error_75, error_50, error_25, line_history, line_forecast ]

    return data


def get_forcase_plot_shapes(history_data, future_data):
    shapes = [
        {
            "type": "rect",
            # x-reference is assigned to the x-values
            "xref": "x",
            "x0": 0,
            "x1": len(history_data),
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
            "x0": len(history_data),
            "x1": len(future_data),
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



def get_series_figure(history_task_num_list,future_task_num_list):

    shapes = get_forcase_plot_shapes(history_task_num_list, future_task_num_list)
    data=get_forecast_plot_data(history_task_num_list,future_task_num_list)

    time_slice_list = []
    for i in range(len(history_task_num_list+future_task_num_list)):
        time_slice_list.append(i)


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
            # type="date",
            range=[
                time_slice_list[-15], # Recent point in history
                time_slice_list[-1]  # End of forecast range
            ],

            rangeselector=dict(
                buttons=list(
                    [
                        dict(
                            count=int(time_slice_list[-1]*0.5),
                            label="50%",
                            step="day",
                            stepmode="backward",


                        ),
                        dict(
                            count=int(time_slice_list[-1]*0.8),
                            label="80%",
                            step="day",
                            stepmode="backward",
                        ),
                        dict(
                            count=time_slice_list[-1],
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
                    time_slice_list[0],  # Recent point in history
                    time_slice_list[-1]  # End of forecast range
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


def update_sparkline(interval, param):
    x_array = state_dict["Batch"]["data"].tolist()
    y_array = state_dict[param]["data"].tolist()

    if interval == 0:
        x_new = y_new = None

    else:
        if interval >= max_length:
            total_count = max_length
        else:
            total_count = interval
        x_new = x_array[:total_count][-1]
        y_new = y_array[:total_count][-1]

    return dict(x=[[x_new]], y=[[y_new]]), [0]


def update_count(interval, col, data):
    if interval == 0:
        return "0", "0.00%", 0.00001, "#92e0d3"

    if interval > 0:

        if interval >= max_length:
            total_count = max_length - 1
        else:
            total_count = interval - 1

        ooc_percentage_f = data[col]["ooc"][total_count] * 100
        ooc_percentage_str = "%.2f" % ooc_percentage_f + "%"

        # Set maximum ooc to 15 for better grad bar display
        if ooc_percentage_f > 15:
            ooc_percentage_f = 15

        if ooc_percentage_f == 0.0:
            ooc_grad_val = 0.00001
        else:
            ooc_grad_val = float(ooc_percentage_f)

        # Set indicator theme according to threshold 5%
        if 0 <= ooc_grad_val <= 5:
            color = "#92e0d3"
        elif 5 < ooc_grad_val < 7:
            color = "#f4d44d"
        else:
            color = "#FF0000"

    return str(total_count + 1), ooc_percentage_str, ooc_grad_val, color

# ======================main layout===================
app.layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        dcc.Interval(
            id="interval-component",
            interval=2 * 1000,  # in milliseconds
            n_intervals=50,  # start at batch 50
            disabled=True,
        ),
        dcc.Interval(
            id="global-data-interval-component",
            interval=2 * 1000,  # in milliseconds
            n_intervals=0,
        ),
        html.Div(
            id="app-container",
            children=[
                build_tabs(),
                # Main app
                html.Div(id="app-content"),
            ],
        ),

        dcc.Store(id="global-data-store"),
        dcc.Store(id="value-setter-store", data=init_value_setter_store()),
        dcc.Store(id="n-interval-stage", data=50),
        # generate_modal(),
    ],
)

@app.callback(
    [Output("app-content", "children"), Output("interval-component", "n_intervals")],
    [Input("app-tabs", "value")],
    [State("n-interval-stage", "data")],
)
def render_tab_content(tab_switch, stopped_interval):
    if tab_switch == "se_tab":
        return build_se_tab(stopped_interval),stopped_interval
    if tab_switch == "rl_tab":
        return build_rl_tab(), stopped_interval

# Update global data according to iteraval
@app.callback(
    [Output("global-data-store","data")],
    [Input("global-data-interval-component","n_intervals")]
)
def update_global_data(n_intervals):
    global_data=gl.get_value("GLOBAL_DATA")
    if global_data[0]!=None:
        global LOWER_SIZE_LIMIT
        global UPPER_SIZE_LIMIT
        LOWER_SIZE_LIMIT = global_data[0]["time_window_condition"]["min_time_windows"][-1]
        UPPER_SIZE_LIMIT = global_data[0]["time_window_condition"]["max_time_windows"][-1]
    return global_data



# ======================simulation environment tab callback===================

# Update interval
@app.callback(
    Output("n-interval-stage", "data"),
    [Input("app-tabs", "value")],
    [
        State("interval-component", "n_intervals"),
        State("interval-component", "disabled"),
        State("n-interval-stage", "data"),
    ],
)
def update_interval_state(tab_switch, cur_interval, disabled, cur_stage):
    if disabled:
        return cur_interval

    if tab_switch == "rl_tab":
        return cur_interval
    return cur_stage


# Callbacks for stopping interval update
@app.callback(
    [Output("interval-component", "disabled"), Output("stop-button", "buttonText")],
    [Input("stop-button", "n_clicks")],
    [State("interval-component", "disabled")],
)
def stop_production(n_clicks, current):
    if n_clicks == 0:
        return True, "start"
    return not current, "stop" if current else "start"


# ======= update progress gauge =========
@app.callback(
    output=Output("progress-gauge", "value"),
    inputs=[Input("interval-component", "n_intervals")],
)
def update_gauge(interval):
    if interval < max_length:
        total_count = interval
    else:
        total_count = max_length

    return int(total_count)

#===========update distribution overview graph==========
@app.callback(
    output=Output("distribution_overview_graph", "figure"),
    inputs=[Input("global-data-store","data")],
)
def update_distribution_overview_graph(global_data):

    data = []
    # # draw distribution overview
    if global_data!=None:
        road_network_edges = global_data["road_network_edges"]
        distribution_data = global_data["distribution_data"]
        worker_remaining_time_list = distribution_data[0]
        worker_total_time_list = distribution_data[1]
        worker_location_list = distribution_data[2]
        task_deadline_list = distribution_data[3]
        task_location_list = distribution_data[4]
        task_transparency_list=distribution_data[5]


        # draw road network
        for i in range(len(road_network_edges)):
            edge = road_network_edges[i]
            edge_x_list = []
            edge_y_list = []
            edge_x_list.append(edge[0][0])
            edge_y_list.append(edge[0][1])
            edge_x_list.append(edge[1][0])
            edge_y_list.append(edge[1][1])
            trace = go.Scatter(
                x=edge_x_list,
                y=edge_y_list,
                line=dict(color="rgba(192,192,192,0.6)"),
                name="road_"+str(i),
                showlegend=False,
            )
            data.append(trace)

        # draw worker location and color
        for i in range(len(worker_location_list)):
            worker_transparency = 1 - (worker_remaining_time_list[i] / worker_total_time_list[i])
            if worker_remaining_time_list[i] == 0:
                trace = go.Scatter(
                    x=[worker_location_list[i][0]],
                    y=[worker_location_list[i][1]],
                    mode="markers",
                    marker_size=10,
                    marker_color="rgba(0,168,141,"+str(worker_transparency)+")",
                    name="Worker_"+str(i),
                    showlegend=False,
                )
                data.append(trace)
            else:
                trace = go.Scatter(
                    x=[worker_location_list[i][0]],
                    y=[worker_location_list[i][1]],
                    mode="markers",
                    marker_size=10,
                    marker_color="rgba(255,127,14," + str(worker_transparency) + ")",
                    name="Worker_" + str(i),
                    showlegend=False,
                )
                data.append(trace)


        # draw task location and color
        for i in range(len(task_location_list)):
            trace = go.Scatter(
                x=[task_location_list[i][0]],
                y=[task_location_list[i][1]],
                mode="markers",
                marker_size=10,
                marker_color="rgba(31,119,180," + str(task_transparency_list[i]) + ")",
                showlegend=False,
                name="Task_" + str(i),
            )
            data.append(trace)

        # draw legend
        trace = go.Scatter(
            x=[50],
            y=[30],
            mode="markers",
            marker_color="rgba(0,168,141,1)",
            marker_size=10,
            name="Free Worker"
        )
        data.append(trace)
        trace = go.Scatter(
            x=[-20],
            y=[30],
            mode="markers",
            marker_color="rgba(255,127,14,1)",
            marker_size=10,
            name="Busy Worker"
        )
        data.append(trace)

        trace = go.Scatter(
            x=[0],
            y=[10],
            mode="markers",
            marker_color="rgba(31,119,180,1)",
            marker_size=10,
            name="Task"
        )
        data.append(trace)

    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={"color": "darkgray"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "xanchor": "center",
            "y": 1,
            "x": 0.5,
        },
        xaxis=dict(
            title="Longitude",
            showline=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Latitude",
            showline=False,
            showgrid=False,
            zeroline=False,
        )

    )
    return go.Figure(
        data=data,
        layout=layout
    )

#===============update traffic overview graph====================
@app.callback(
    output=Output("traffic_overview_graph", "figure"),
    inputs=[Input("global-data-store","data")],
)
def update_traffic_overview_graph(global_data):
    data = []
    # # draw traffic overview
    if global_data!=None:
        road_network_edges = global_data["road_network_edges"]
        edge_colors = global_data["edge_colors"]
        for i in range(len(road_network_edges)):
            edge = road_network_edges[i]
            edge_x_list = []
            edge_y_list = []
            edge_x_list.append(edge[0][0])
            edge_y_list.append(edge[0][1])
            edge_x_list.append(edge[1][0])
            edge_y_list.append(edge[1][1])
            road_conditions="Smooth"
            if edge_colors[i]=="orange":
                road_conditions="Slow"
            if edge_colors[i] == "red":
                road_conditions = "Congested"
            trace = go.Scatter(
                x=edge_x_list,
                y=edge_y_list,
                showlegend=False,
                line=dict(color=edge_colors[i]),
                name=road_conditions

            )
            data.append(trace)

        # draw legend
        trace = go.Scatter(
            x=[-10,-10],
            y=[-10,-10],
            line=dict(color="#057748"),
            mode="lines",
            name="Smooth"
        )
        data.append(trace)
        trace = go.Scatter(
            x=[-10, -10],
            y=[-10, -10],
            mode="lines",
            line=dict(color="#ffa631"),
            name="Slow"
        )
        data.append(trace)
        trace = go.Scatter(
            x=[-10, -10],
            y=[-10, -10],
            mode="lines",
            line=dict(color="#ff3300"),
            name="Congested"
        )
        data.append(trace)


    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={"color": "darkgray"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "xanchor": "center",
            "y": 1,
            "x": 0.5,
        },

        xaxis=dict(
            title="Longitude",
            showline=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Latitude",
            showline=False,
            showgrid=False,
            zeroline=False,
        )

    )
    return go.Figure(
        data=data,
        layout=layout
    )

# decorator for list of output
def create_callback(param):
    def callback(interval, stored_data):
        count, ooc_n, ooc_g_value, indicator = update_count(
            interval, param, stored_data
        )
        spark_line_data = update_sparkline(interval, param)
        return count, spark_line_data, ooc_n, ooc_g_value, indicator

    return callback


for param in params[1:]:
    update_param_row_function = create_callback(param)
    app.callback(
        output=[
            Output(param + suffix_count, "children"),
            Output(param + suffix_sparkline_graph, "extendData"),
            Output(param + suffix_ooc_n, "children"),
            Output(param + suffix_ooc_g, "value"),
            Output(param + suffix_indicator, "color"),
        ],
        inputs=[Input("interval-component", "n_intervals")],
        state=[State("value-setter-store", "data")],
    )(update_param_row_function)



# =============Update Worker Condition =============
@app.callback(
    output=Output("metric-rows", "children"),
    inputs=Input("global-data-store","data")
)
def update_worker_condition_graph(global_data):

    div_children=[]

    if global_data!=None:
        worker_condition_list=global_data["worker_condition"]
        for worker_condition in worker_condition_list:

            time_slice_list=[]
            for i in range(len(worker_condition["Speed"])):
                time_slice_list.append(i)


            item="worker_"+str(worker_condition["ID"])+"_condition"
            div_id=item+"_div"
            speed_graph_id = item + "_speed_graph"
            name_id = item + "_name"
            task_number_id = item + "_task_number"
            deadline_graph_id = item + suffix_ooc_g
            state_id = item + "_state_flag"

            one_worker_condition=\
                generate_metric_row(
                    div_id,
                    None,
                    {
                        "id": item+"_ID",
                        # "className": "metric-row-button-text",
                        "children": worker_condition["ID"]
                    },
                    {"id": name_id, "children": worker_condition["Name"]},
                    {
                        "id": item + "_sparkline",
                        "children": dcc.Graph(
                            id=speed_graph_id,
                            style={"width": "100%", "height": "95%"},
                            config={
                                "staticPlot": False,
                                "editable": False,
                                "displayModeBar": False,
                            },
                            figure=go.Figure(
                                {
                                    "data": [
                                        {
                                            "x":time_slice_list,
                                            "y": worker_condition["Speed"],
                                            "mode": "lines+markers",
                                            "name": item,
                                            "line": {"color": "#f4d44d"},
                                        }
                                    ],
                                    "layout": {
                                        "uirevision": True,
                                        "margin": dict(l=0, r=0, t=4, b=4, pad=0),
                                        "xaxis": dict(
                                            showline=False,
                                            showgrid=False,
                                            zeroline=False,
                                            showticklabels=False,
                                        ),
                                        "yaxis": dict(
                                            showline=False,
                                            showgrid=False,
                                            zeroline=False,
                                            showticklabels=False,
                                        ),
                                        "paper_bgcolor": "rgba(0,0,0,0)",
                                        "plot_bgcolor": "rgba(0,0,0,0)",
                                    },
                                }
                            ),
                        ),
                    },
                    {"id":  task_number_id, "children": worker_condition["TaskNumber"]},
                    {
                        "id": deadline_graph_id + "_container",
                        "children": daq.GraduatedBar(
                            id=deadline_graph_id,
                            color={
                                "gradient": True,
                                "ranges": {
                                    "#ff3300": [0, 2],
                                    "#ffc773 ": [2, 7],
                                    "#057748": [7, 10],
                                }
                            },
                            showCurrentValue=False,
                            # max=300,
                            theme="dark",
                            value=int(worker_condition["Deadline"]/30),

                        ),

                    },
                    {
                        "id": item + "_state",
                        "children": daq.Indicator(
                            id=state_id, value=True, color=worker_condition["State"], size=12
                        ),
                    },
                )

            div_children.append(one_worker_condition)
    return div_children




# =============Update Worker Revenue==============
@app.callback(
    output=Output("worker_revenue_piechart", "figure"),
    inputs=[Input("global-data-store","data")],
)
def update_worker_revenue_pie(global_data):
    if global_data == None:
        return {
            "data": [],
            "layout": {
                "font": {"color": "white"},
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
            },
        }

    worker_revenue_list=[]
    worker_name_list=[]
    color_list=[]
    for worker_condition in  global_data["worker_condition"]:
        worker_revenue_list.append(worker_condition["Revenue"])
        worker_name_list.append(worker_condition["Name"])

    for worker_condition in global_data["worker_condition"]:
        if worker_condition["Revenue"]>sum(worker_revenue_list)/len(worker_revenue_list):
            color_list.append("#f45060")
        else:
            color_list.append("#91dfd2")

    new_figure = {
        "data": [
            {
                "labels": worker_name_list,
                "values": worker_revenue_list,
                "type": "pie",
                "marker": {"colors": color_list, "line": dict(color="white", width=2)},
                "hoverinfo": "label",
                "textinfo": "label",
            }
        ],
        "layout": {
            "margin": dict(t=20, b=50),
            "uirevision": True,
            "font": {"color": "white"},
            "showlegend": False,
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "autosize": True,
        },
    }
    return new_figure


#  ======= Update weather conditions graph ============
@app.callback(
    output=Output("weather-condition-chart", "figure"),
    inputs=Input("global-data-store","data")
)
def update_weather_condition_graph(global_data):

    weather_data = []
    precipitation_list = []
    wind_intensity_list = []
    fog_density_list = []
    time_slice_list = []

    # weather level
    # w_excellent = 10
    # w_good = 25
    # w_medium = 40
    # w_poor = 55
    w_excellent = 5
    w_good = 15
    w_medium = 25
    w_poor = 40

    evs_trace = {
        "x": [],
        "y": [],
        "name": "Extreme Value",
        "mode": "markers",
        "marker": dict(color="rgba(210, 77, 87, 0.7)", symbol="square", size=11),
    }

    if global_data != None:
        weather_data = global_data["weather"]
        precipitation_list = weather_data[0]
        wind_intensity_list = weather_data[1]
        fog_density_list = weather_data[2]
        for i in range(len(wind_intensity_list)):

            time_slice_list.append(i)
            # Collect extreme values
            if precipitation_list[i] > w_poor:
                evs_trace["x"].append(i)
                evs_trace["y"].append(precipitation_list[i])
            if wind_intensity_list[i] > w_poor:
                evs_trace["x"].append(i)
                evs_trace["y"].append(wind_intensity_list[i])

            if fog_density_list[i] > w_poor:
                evs_trace["x"].append(i)
                evs_trace["y"].append(fog_density_list[i])

    histo_trace = {
        # "x": time_slice_list,
        "y": precipitation_list+ wind_intensity_list + fog_density_list,
        "type": "histogram",
        "orientation": "h",
        "name": "Distribution",
        "xaxis": "x2",
        "yaxis": "y2",
        "marker": {"color": "#f4d44d"},

    }

    fig = {
        "data": [
            {
                "x": time_slice_list,
                "y": precipitation_list,
                "mode": "lines+markers",
                "name": "Precipitation",
                "line": {"color": "rgba(255,127,14,0.8)"},
            },
            {
                "x": time_slice_list,
                "y": fog_density_list,
                "mode": "lines+markers",
                "name": "Fog",
                "line": {"color": "#999999"},
            },
            {
                "x": time_slice_list,
                "y": wind_intensity_list,
                "mode": "lines+markers",
                "name": "Wind",
                "line": {"color": "#0C5AA6"},
            },

            evs_trace,
            histo_trace,
        ]
    }

    len_figure = len(fig["data"][0]["x"])

    fig["layout"] = dict(
        margin=dict(t=40),
        hovermode="closest",
        uirevision="weather",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"font": {"color": "darkgray"}, "orientation": "h", "x": 0, "y": 1.1},
        font={"color": "darkgray"},
        showlegend=True,
        xaxis={
            "zeroline": False,
            "showgrid": False,
            "title": "Time Slice",
            "showline": False,
            "domain": [0, 0.8],
            "titlefont": {"color": "darkgray"},
        },
        yaxis={
            "title": "Climate Indicator",
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "autorange": True,
            "titlefont": {"color": "darkgray"},
        },
        annotations=[
            {
                "x": 0.75,
                "y": w_excellent,
                "xref": "paper",
                "yref": "y",
                "text": "Excellent:" + str(w_excellent),
                "showarrow": False,
                "font": {"color": "white"},
            },
            {
                "x": 0.75,
                "y": w_good,
                "xref": "paper",
                "yref": "y",
                "text": "Good: " + str(w_good),
                "showarrow": False,
                "font": {"color": "white"},
            },
            {
                "x": 0.75,
                "y": w_medium,
                "xref": "paper",
                "yref": "y",
                "text": "Medium: " + str(w_medium),
                "showarrow": False,
                "font": {"color": "white"},
            },
            {
                "x": 0.75,
                "y": w_poor,
                "xref": "paper",
                "yref": "y",
                "text": "Poor: " + str(w_poor),
                "showarrow": False,
                "font": {"color": "white"},
            },

        ],
        shapes=[
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": 0,
                "y0": w_excellent,
                "x1": len_figure ,
                "y1": w_excellent,
                "line": {"color": "#91dfd2", "width": 1, "dash": "dot"},
            },
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": 0,
                "y0": w_good,
                "x1": len_figure ,
                "y1": w_good,
                "line": {"color": "#91dfd2", "width": 1, "dash": "dot"},
            },
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": 0,
                "y0": w_medium,
                "x1": len_figure ,
                "y1": w_medium,
                "line": {"color": "rgb(255,127,80)", "width": 1, "dash": "dot"},
            },
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": 0,
                "y0": w_poor,
                "x1": len_figure ,
                "y1": w_poor,
                "line": {"color": "rgba(210, 77, 87,1)", "width": 2},
            }
        ],
        xaxis2={
            "title": "Count",
            "domain": [0.8, 1],  # 70 to 100 % of width
            "titlefont": {"color": "darkgray"},
            "showgrid": False,
        },
        yaxis2={
            "anchor": "free",
            "overlaying": "y",
            "side": "right",
            "showticklabels": False,
            "titlefont": {"color": "darkgray"},
        },
    )

    return fig


#===================Update future task graph=====================

@app.callback(
    output=[ Output("button-history-tasks-12", "type"),
             Output("button-history-tasks-15", "type"),
             Output("button-history-tasks-24", "type"),
             Output("button-history-tasks-36", "type"),
             Output("button-history-tasks-48", "type"),
             Output("button-history-tasks-12", "nClicks"),
             Output("button-history-tasks-15", "nClicks"),
             Output("button-history-tasks-24", "nClicks"),
             Output("button-history-tasks-36", "nClicks"),
             Output("button-history-tasks-48", "nClicks")
             ],
    inputs=[ Input("button-history-tasks-12", "nClicks"),
             Input("button-history-tasks-15", "nClicks"),
             Input("button-history-tasks-24", "nClicks"),
             Input("button-history-tasks-36", "nClicks"),
             Input("button-history-tasks-48", "nClicks")
             ],
    prevent_initial_call=True

)
def update_history_task_length(n_clicks_12,n_clicks_15,n_clicks_24,n_clicks_36,n_clicks_48):
    # You can connect your own prediction methods and return prediction figure

    if n_clicks_12!=None:
        return "primary",None,None,None,None,None,None,None,None,None
    if n_clicks_15!=None:
        return None,"primary", None,None,None,None,None,None,None,None
    if n_clicks_24!=None:
        return None,None,"primary",None,None, None,None,None,None,None
    if n_clicks_36!=None:
        return None,None,None, "primary", None, None,None,None,None,None
    if n_clicks_48!=None:
        return None,None,None,None,"primary", None,None,None,None,None


@app.callback(
    output= Output("forcast_graph", "children"),
    inputs=[ Input("global-data-store", "data")],

)

def update_forcast_graph(global_data):


    history_task_num_list=[]
    future_task_num_list=[]
    if global_data != None:
        history_task_num_list=global_data["task_num"][:-5]
        future_task_num_list=global_data["task_num"][-5:]

    if len(history_task_num_list) > 15:
        series_figure = get_series_figure(history_task_num_list,future_task_num_list)

        forcast_graph = dcc.Graph(
            figure=series_figure,
            config={
                # "modeBarButtonsToRemove": [
                #     "sendDataToCloud",
                #     "autoScale2d",
                #     "hoverClosestCartesian",
                #     "hoverCompareCartesian",
                #     "lasso2d",
                #     "select2d",
                #     "toggleSpikelines",
                # ],
                "displaylogo": False,
            },
        )
    else:
        description_text="Accumulating historical data, please wait for "+ str(15-len(history_task_num_list)) +" seconds."
        forcast_graph = fac.AntdEmpty(description=description_text, style={"margin-top": "80px","color":"darkgray"})

    return forcast_graph



# ======================rl decision tab callback==========================

# ===== Callbacks to update values based on store data and dropdown selection =====
@app.callback(
    output=[
        Output("value-setter-panel", "children"),
        Output("ud_lsl_input", "value"),
        Output("ud_usl_input", "value"),

    ],
    inputs=[
            Input("value-setter-reset-btn", "n_clicks"),
            ],
    state=[
        State("usl-value", "children"),
        State("lsl-value", "children")
          ],
)
def reset_size_value_setter(n_clicks,usl_current_value,lsl_current_value):

    if n_clicks==0:
        return (
            [
                build_value_setter_line(
                    "value-setter-panel-header",
                    "header-value",
                    "Specs",
                    "Current Value",
                    "Set new value",
                ),
                build_value_setter_line(
                    "value-setter-panel-lsl",
                    "lsl-value",
                    "Lower Size limit",
                    LOWER_SIZE_LIMIT,
                    ud_lsl_input,
                ),
                build_value_setter_line(
                    "value-setter-panel-usl",
                    "usl-value",
                    "Upper Size limit",
                    UPPER_SIZE_LIMIT,
                    ud_usl_input,
                ),

            ],
            LOWER_SIZE_LIMIT,
            UPPER_SIZE_LIMIT,

        )
    else:
        return (
            [
                build_value_setter_line(
                    "value-setter-panel-header",
                    "header-value",
                    "Specs",
                    "Current Value",
                    "Set new value",
                ),
                build_value_setter_line(
                    "value-setter-panel-lsl",
                    "lsl-value",
                    "Lower Size limit",
                    lsl_current_value,
                    ud_lsl_input,
                ),
                build_value_setter_line(
                    "value-setter-panel-usl",
                    "usl-value",
                    "Upper Size limit",
                    usl_current_value,
                    ud_usl_input,
                ),
            ],
            lsl_current_value,
            usl_current_value,

        )



# ====== Callbacks to update time window size via click =====
@app.callback(
    output=[
        Output("usl-value", "children"),
        Output("lsl-value", "children"),
        Output('notification-div', 'children'),
    ],
    inputs=[Input("value-setter-update-btn", "n_clicks")],
    state=[
        State("ud_usl_input", "value"),
        State("ud_lsl_input", "value"),
        State("usl-value", "children"),
        State("lsl-value", "children"),
    ],
)
def update_size_value_setter(n_clicks, usl_new_value, lsl_new_value,usl_current_value,lsl_current_value):
        if n_clicks!=None:
            if usl_new_value>lsl_new_value:
                if lsl_new_value<LOWER_WINDOW_SIZE or lsl_new_value>UPPER_WINDOW_SIZE:
                    return [
                        usl_current_value,
                        lsl_current_value,
                        fac.AntdNotification(
                            message='Update Failed',
                            description='The value range of "Lower Size limit" is ['+str(LOWER_WINDOW_SIZE)+","+str(UPPER_WINDOW_SIZE)+']',
                            type="error"
                        )]
                if usl_new_value < LOWER_WINDOW_SIZE or usl_new_value > UPPER_WINDOW_SIZE:
                    return [
                        usl_current_value,
                        lsl_current_value,
                        fac.AntdNotification(
                            message='Update Failed',
                            description='The value range of "Upper Size limit" is [0,50]',
                            type="error"
                        )]

                global_data= gl.get_value("GLOBAL_DATA")
                if global_data[0]!=None:
                    global_data[0]["new_max_time_window"] = usl_new_value
                    global_data[0]["new_min_time_window"] = lsl_new_value
                    gl.set_value("GLOBAL_DATA", global_data)
                return [
                    usl_new_value,
                    lsl_new_value,
                    fac.AntdNotification(
            message='Updatet Successful',
            type="success"
        )]
            else:
                return [
                usl_current_value,
                lsl_current_value,
                fac.AntdNotification(
            message='Update Failed',
            description='"Upper Size limit" should be greater than "Lower Size limit"',
            type="error"
        )]
        else:
            return [
                usl_current_value,
                lsl_current_value,
                None
            ]


# ====== Callbacks to update time window figure =====

@app.callback(
    Output("time-window", "figure"), [Input("global-data-store","data")]
)
def gen_time_window_figure(global_data):

    if global_data!=None:
        time_window_condition = global_data["time_window_condition"]
        max_time_window_list=time_window_condition["max_time_windows"]
        min_time_window_list = time_window_condition["min_time_windows"]
        current_time_window_list=time_window_condition["current_time_windows"]
        # max_error=[]
        # min_error=[]
        step_list=[]

        # print("cu:",current_time_window_list)
        # print("max:",max_time_window_list)
        # print("min",min_time_window_list)
        for i in range(len(current_time_window_list)):
            step_list.append(i)
        #     max_error.append(max_time_window_list[i]-current_time_window_list[i])
        #     min_error.append(current_time_window_list[i]-min_time_window_list[i])

        trace = dict(
            type="scatter",
            x=step_list,
            y=current_time_window_list,
            line={"color": "rgba(46,169,223,1)"},
               #42C4F7
            width = 5,
            thickness=1.5,
            hoverinfo="skip",
            # showlegend=False,
            # error_y={
            #     "type": "data",
            #     "array": max_error,
            #     "thickness": 1.5,
            #     "width": 2,
            #     "color": "#B4E8FC",
            # },
            name="window size",
            mode="lines",
        )

        trace_list=[]

        range_trace = dict(
            x=[0, 0],
            y=[min_time_window_list[0], max_time_window_list[0]],
            mode="markers+lines",
            thickness=1,
            width=1,
            # showlegend=False,
            name="window range",
            line=dict(color='rgba(186,203,219,0.5)')
        )
        trace_list.append(range_trace)

        for i in range(len(max_time_window_list)):

            range_trace=dict(
                x=[i,i],
                y=[min_time_window_list[i],max_time_window_list[i]],
                mode="markers+lines",
                thickness = 1,
                width = 1,
                showlegend=False,
                line=dict(color='rgba(186,203,219,0.5)')
            )
            trace_list.append(range_trace)

        trace_list.append(trace)


        layout = dict(
            plot_bgcolor=app_color["rl_graph_bg"],
            paper_bgcolor=app_color["rl_graph_bg"],
            font={"color": "#fff"},
            height=700,
            legend={ "orientation": "h", "x": 0, "y": 1.1},
            xaxis={
                "range": [0, 120],
                "showline": True,
                "zeroline": False,
                # "fixedrange": True,
                "tickvals": [0, 30, 60, 90, 120],
                "ticktext": ["0","30","60","90","120"],
                "title": "Step",
            },
            yaxis={
                "range": [
                    min(0, min(current_time_window_list)),
                    max(35, max(current_time_window_list) + (max(max_time_window_list)-sum(current_time_window_list)/len(current_time_window_list))),
                ],
                "showgrid": True,
                "showline": True,
                # "fixedrange": True,
                "zeroline": False,
                "gridcolor": app_color["rl_graph_line"],
                # "nticks": max(6, round(current_time_window_list.iloc[-1] / 10)),
            },
        )

        return dict(data=trace_list, layout=layout)

    else:
        return dict(data=[], layout=None)





@app.callback(
    [
     Output("window-train", "figure"),
     Output("train-step-text","children"),
     Output("train-step-percent","percent")
     ],
    inputs=[Input("global-data-store","data")],
)
def gen_revenue_loss_histogram(global_data):
    """
    Genererate window revenue and loss histogram graph.
    """
    if global_data!=None:

        time_window_condition = global_data["time_window_condition"]
        total_train_step=time_window_condition["total_train_step"]
        current_train_step=time_window_condition["current_train_step"]
        loss_list=time_window_condition["loss"]
        time_window_revenue_list=time_window_condition["time_window_revenues"]
        window_step_list=[]
        loss_step_list=[]
        for i in range(len(time_window_revenue_list)):
            window_step_list.append(i)
        for i in range(len(loss_list)):
            loss_step_list.append(i)

        if len(time_window_revenue_list)!=0:
            avg_val = float(sum(time_window_revenue_list)) / len(time_window_revenue_list)
            median_val = np.median(time_window_revenue_list)
        else:
            avg_val=0
            median_val=0

        revenue_trace = dict(
            type="bar",
            x=window_step_list,
            y=time_window_revenue_list,
            marker={"color": app_color["rl_graph_line"]},
            showlegend=False,
            hoverinfo="x+y",
        )

        average_scatter=dict(
            type="scatter",
            x=window_step_list,
            y=[avg_val],
            mode="lines",
            line={"dash": "dash", "color": "#2E5266"},
            marker={"opacity": 0},
            visible=True,
            name="Average",
        )

        median_scatter = dict(
            type="scatter",
            x=window_step_list,
            y=[median_val],
            mode="lines",
            line={"dash": "dot", "color": "#BD9391"},
            marker={"opacity": 0},
            visible=True,
            name="Median",
        )


        loss_trace = dict(
            type="scatter",
            mode="lines",
            line={"color": "#42C4F7"},
            y=loss_list,
            x=loss_step_list,
            name="Loss",
            xaxis= "x1",
            yaxis= "y2",
        )

        layout = dict(
            height=350,
            plot_bgcolor=app_color["rl_graph_bg"],
            paper_bgcolor=app_color["rl_graph_bg"],
            font={"color": "#fff"},
            xaxis={
                "title": "Step",
                "showgrid": False,
                "showline": False,
                "fixedrange": True,
            },
            yaxis={
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "title": "Window Revenue",
                "fixedrange": True,
            },
            yaxis2={
                "title":"Loss",
                "showgrid": False,
                "showline": False,
                "side": "right",

            },
            autosize=True,
            bargap=0.01,
            bargroupgap=0,
            hovermode="closest",
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "xanchor": "center",
                "y": 1,
                "x": 0.5,
            },
            shapes=[
                {
                    "xref": "x",
                    "yref": "y",
                    "x0": 0,
                    "y0": avg_val,
                    "x1": len(window_step_list),
                    "y1": avg_val,
                    "type": "line",
                    "line": {"dash": "dash", "color": "#2E5266", "width": 5},
                },
                {
                    "xref": "x",
                    "yref": "y",
                     "x0": 0,
                    "y0": median_val,
                    "x1": len(window_step_list),
                    "y1": median_val,
                    "type": "line",
                    "line": {"dash": "dot", "color": "#BD9391", "width": 5},
                },
            ],
        )

        #train step progress
        train_step=str(current_train_step)+"/"+str(total_train_step)

        return dict(data=[revenue_trace, average_scatter, median_scatter, loss_trace], layout=layout),train_step,current_train_step/(total_train_step/100)
    else:
        return dict(data=[], layout=[]),"0/0",0




# Running the server
# if __name__ == "__main__":
#     app.run_server(debug=True, port=8050)
