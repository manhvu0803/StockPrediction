import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from binance.client import Client
import stock_model

PAST_CHANGE_COUNT = 9

CurrentModel = 0

Models = [
    ["BTC-USD.csv.keras", "BTC-USD.csv_roc.keras", 60],
    ["XGBoost_model.json", "XGBoost_Roc_model.json", 21],
    ["RNN_model.keras", "RNN_Roc_model.keras", 50],
]

app = dash.Dash()
server = app.server
client = Client()

app.layout = html.Div(
    [
        html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
        html.H2("Model", style={"textAlign": "center"}),
        dcc.Dropdown(
            id='model-dropdown',
            options=[
                {"label": "LGTM", "value": 0},
                {"label": "XGBoost", "value": 1},
                {"label": "RNN", "value": 2},
            ],
            value=0,
            style={"display": "block", "width": "60%", "margin-left": "auto", "margin-right": "auto"}
        ),
        html.Div(style={"min-height": "1em"}),
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Candle chart",
                    children=[dcc.Graph(id='highlow', style={"height": 800})]
                ),
                dcc.Tab(
                    label="Close price",
                    children=[dcc.Graph(id='close', style={"height": 600})]
                ),
                dcc.Tab(
                    label="Rate of change",
                    children=[dcc.Graph(id='roc', style={"height": 600})]
                )
            ]
        ),
        dcc.Interval(
            id="update-interval",
            interval=60 * 1000
        ),
    ],
)

@app.callback(
    Output('highlow', 'figure'),
    Output('close', 'figure'),
    Output('roc', 'figure'),
    Input('update-interval', 'n_intervals'),
    Input('model-dropdown', 'value')
)
def update_graph(interval, modelIndex):
    print("Update graph")

    global CurrentModel
    CurrentModel = modelIndex

    df = pd.DataFrame(
        data=client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=1000 ),
        columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trade_count", "buy_base", "buy_take", "ignore"]
    )
    df["open"] = pd.to_numeric(df["open"])
    df["high"] = pd.to_numeric(df["open"])
    df["low"] = pd.to_numeric(df["low"])
    df["close"] = pd.to_numeric(df["close"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df.set_index("open")

    candleFig = go.Candlestick(
        x=df["close_time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="candle",
    )

    closeScatter = go.Scatter(
        x=df["close_time"],
        y=df["close"],
        mode='lines',
        opacity=0.7,
        name=f'Close price',
        textposition='bottom center'
    )

    return createFigure([candleFig, closeScatter], yTitle="Price (USD)", height=800), closeGraph(df), rocGraph(df)

def closeGraph(df):
    closeScatter = go.Scatter(
        x=df["close_time"],
        y=df["close"],
        mode='lines',
        opacity=0.7,
        name=f'Close price',
        textposition='bottom center'
    )

    train, valid = stock_model.predict(df, Models[CurrentModel][0], shape=Models[CurrentModel][2], timeColumn="close_time", isXgb=CurrentModel==1)
    predictScatter = go.Scatter(
        x=valid.index,
        y=valid["predictions"],
        name="Predicted close price",
        textposition='bottom center'
    )
    return createFigure([closeScatter, predictScatter], yTitle="Price (USD)")

def rocGraph(df):
    roc = []

    for i in range(PAST_CHANGE_COUNT, len(df)):
        currentData = df.iloc[i]
        pastData = df.iloc[i - PAST_CHANGE_COUNT]
        roc.append(currentData["close"] / pastData["close"] * 100)

    df = df.loc[PAST_CHANGE_COUNT:]
    df.loc[:, "close"] = roc

    train, valid = stock_model.predict(df, Models[CurrentModel][1], shape=Models[CurrentModel][2], timeColumn="close_time", isXgb=CurrentModel==1)
    predictScatter = go.Scatter(
        x=valid.index,
        y=valid["predictions"],
        name="Predicted ROC",
        textposition='bottom center'
    )

    rocScatter = go.Scatter(
        x=df.iloc[PAST_CHANGE_COUNT:]["close_time"],
        y=roc,
        mode='lines',
        opacity=0.7,
        name=f'ROC {PAST_CHANGE_COUNT}',
        textposition='bottom center'
    )

    return createFigure([rocScatter, predictScatter], yTitle="%")

def createFigure(data, title="", yTitle="", height=600):
    return {
        'data': data,
        'layout': go.Layout(
            height=height,
            title=title,
            xaxis={
                "title":"Date",
                'rangeslider': {'visible': True},
                'type': 'date'
            },
            yaxis={"title": yTitle},
            plot_bgcolor='#D5FFFF',
            paper_bgcolor='#D5FFFF',
        )
    }

if __name__ == '__main__':
    app.run_server(debug=True)