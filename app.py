
from flask import Flask, render_template, jsonify, request
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os
import requests
import utils

app = Flask(__name__)

FALLBACK_USD_INR = 85.0
EXCHANGE_RATE_URL = "https://open.er-api.com/v6/latest/USD"


def get_usd_inr_rate():
    """Fetch the latest USD/INR exchange rate."""
    try:
        response = requests.get(EXCHANGE_RATE_URL, timeout=4)
        response.raise_for_status()

        payload = response.json()
        rate = float(payload["rates"]["INR"])

        if rate > 0:
            app.config["USD_INR_RATE"] = rate
            return rate

    except (requests.RequestException, ValueError, KeyError, TypeError):
        pass

    return app.config.get("USD_INR_RATE", FALLBACK_USD_INR)


def to_inr(usd_value, rate=None):
    """Convert USD to INR."""
    try:
        value = float(usd_value)

        if pd.isna(value):
            return None

        rate = rate if rate is not None else get_usd_inr_rate()
        return value * rate

    except (TypeError, ValueError):
        return None


def dual_currency(usd_value, rate=None):
    """Format an amount in both USD and INR."""
    try:
        usd_value = float(usd_value)

        if pd.isna(usd_value):
            return "N/A"

        rate = rate if rate is not None else get_usd_inr_rate()
        inr_value = usd_value * rate

        return f"${usd_value:,.2f} / ₹{inr_value:,.2f}"

    except (TypeError, ValueError):
        return "N/A"


@app.route('/')
def home():
    return render_template('index.html', page='home')


@app.route('/eda')
def eda():
    refresh = request.args.get('refresh', '0') == '1'
    data = utils.fetch_bitcoin_data(force_refresh=refresh)

    plots = {}
    dataset_preview = None

    if data is not None:
        # Bitcoin price trend
        fig_price = px.line(
            data,
            x='date',
            y='close',
            title="Bitcoin Price Trend (USD)"
        )

        fig_price.update_layout(
            template="plotly_dark",
            font=dict(family="Inter, sans-serif"),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.05)',
                tickprefix='$',
                separatethousands=True
            )
        )

        fig_price.update_traces(
            hovertemplate=(
                "Date: %{x}<br>"
                "USD: $%{y:,.2f}"
                "<extra></extra>"
            )
        )

        plots['price'] = json.dumps(
            fig_price,
            cls=plotly.utils.PlotlyJSONEncoder
        )

        # Trading volume
        fig_volume = px.area(
            data,
            x='date',
            y='volume',
            title="Bitcoin Trading Volume Over Time"
        )

        fig_volume.update_layout(
            template="plotly_dark",
            font=dict(family="Inter, sans-serif"),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )

        plots['volume'] = json.dumps(
            fig_volume,
            cls=plotly.utils.PlotlyJSONEncoder
        )

        # Dataset statistics
        stats = data.describe().to_html(
            classes="table table-dark table-striped",
            float_format="%.2f"
        )

        # Dataset preview
        cols = [
            'date', 'open', 'high', 'low',
            'close', 'adj_close', 'volume'
        ]

        available_cols = [
            col for col in cols if col in data.columns
        ]

        dataset_preview = data[available_cols].tail(50).to_html(
            classes=(
                "table table-dark table-striped "
                "table-hover"
            ),
            float_format="%.2f",
            index=False
        )

        seasonality = utils.get_seasonality_data()

    else:
        stats = None
        seasonality = None

    return render_template(
        'eda.html',
        page='eda',
        plots=plots,
        stats=stats,
        dataset_preview=dataset_preview,
        seasonality=seasonality
    )


@app.route('/live')
def live_price():
    return render_template('live_price.html', page='live')


@app.route('/api/live_price')
def get_live_price_api():
    usd, inr = utils.get_live_price()
    rate = get_usd_inr_rate()

    return jsonify({
        'usd': usd,
        'inr': inr,
        'usd_formatted': dual_currency(usd, rate),
        'exchange_rate': rate
    })


@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    result = None

    if request.method == 'POST':
        result = utils.train_xgb_model()

        if result and 'error' not in result:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                y=result['y_test'],
                mode='lines',
                name='Actual'
            ))

            fig.add_trace(go.Scatter(
                y=result['predictions'],
                mode='lines',
                name='Predicted'
            ))

            fig.update_layout(
                title="Actual vs Predicted (Test Data)",
                template="plotly_dark",
                font=dict(family="Inter, sans-serif")
            )

            result['plot'] = json.dumps(
                fig,
                cls=plotly.utils.PlotlyJSONEncoder
            )

    return render_template(
        'analysis.html',
        page='analysis',
        result=result
    )


@app.route('/prediction', methods=['GET'])
def prediction():
    try:
        days = int(request.args.get('days', 30))
    except (TypeError, ValueError):
        days = 30

    days = max(1, min(days, 365))

    prediction_data = utils.predict_next_days(days)

    plot = None
    plot_obj = None
    metrics = None

    exchange_rate = get_usd_inr_rate()

    if prediction_data:
        history_dates = prediction_data.get(
            'history_dates', []
        )
        history_prices = prediction_data.get(
            'history_prices', []
        )
        forecast_dates = prediction_data.get(
            'dates', []
        )
        forecast_prices = prediction_data.get(
            'prices', []
        )

        fig = go.Figure()

        # Historical prices
        history_inr = [
            to_inr(price, exchange_rate)
            for price in history_prices
        ]

        fig.add_trace(go.Scatter(
            x=history_dates,
            y=history_prices,
            customdata=history_inr,
            mode='lines',
            name='Historical',
            line=dict(
                color='#888888',
                width=2
            ),
            hovertemplate=(
                "Date: %{x}<br>"
                "USD: $%{y:,.2f}<br>"
                "INR: ₹%{customdata:,.2f}"
                "<extra></extra>"
            )
        ))

        # Predicted prices
        if history_prices and history_dates:
            pred_x = (
                [history_dates[-1]]
                + list(forecast_dates)
            )

            pred_y = (
                [history_prices[-1]]
                + list(forecast_prices)
            )

            pred_inr = [
                to_inr(price, exchange_rate)
                for price in pred_y
            ]

            fig.add_trace(go.Scatter(
                x=pred_x,
                y=pred_y,
                customdata=pred_inr,
                mode='lines+markers',
                name='Predicted',
                line=dict(
                    color='#f7931a',
                    width=3,
                    dash='dash'
                ),
                hovertemplate=(
                    "Date: %{x}<br>"
                    "USD: $%{y:,.2f}<br>"
                    "INR: ₹%{customdata:,.2f}"
                    "<extra></extra>"
                )
            ))

        fig.update_layout(
            title=(
                "Bitcoin Price Prediction: "
                "History & Forecast"
            ),
            template="plotly_dark",
            hovermode="x unified",
            font=dict(family="Inter, sans-serif"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            yaxis=dict(
                title="Price (USD)",
                tickprefix='$',
                separatethousands=True
            ),
            annotations=[dict(
                text=(
                    f"Exchange rate: "
                    f"$1 = ₹{exchange_rate:,.2f}"
                ),
                xref="paper",
                yref="paper",
                x=1,
                y=1.12,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                font=dict(
                    size=12,
                    color="#bbbbbb"
                )
            )]
        )

        plot = json.dumps(
            fig,
            cls=plotly.utils.PlotlyJSONEncoder
        )

        plot_obj = json.loads(plot)

        current_price = prediction_data.get(
            'current_price'
        )

        next_day_price = prediction_data.get(
            'next_day_price'
        )

        price_diff = prediction_data.get(
            'price_diff'
        )

        diff_percent = prediction_data.get(
            'diff_percent'
        )

        metrics = {
            # Original numeric values
            'current_price': current_price,
            'next_day_price': next_day_price,
            'price_diff': price_diff,
            'diff_percent': diff_percent,

            # Formatted USD and INR values
            'current_price_display': dual_currency(
                current_price, exchange_rate
            ),

            'next_day_price_display': dual_currency(
                next_day_price, exchange_rate
            ),

            'price_diff_display': dual_currency(
                price_diff, exchange_rate
            ),

            # INR numeric values
            'current_price_inr': to_inr(
                current_price, exchange_rate
            ),

            'next_day_price_inr': to_inr(
                next_day_price, exchange_rate
            ),

            'price_diff_inr': to_inr(
                price_diff, exchange_rate
            ),

            'exchange_rate': exchange_rate,

            'exchange_rate_display': (
                f"$1 = ₹{exchange_rate:,.2f}"
            )
        }

    return render_template(
        'prediction.html',
        page='prediction',
        plot=plot,
        plot_obj=plot_obj,
        days=days,
        metrics=metrics,
        exchange_rate=exchange_rate,
        exchange_rate_display=(
            f"$1 = ₹{exchange_rate:,.2f}"
        )
    )


@app.route('/about')
def about():
    return render_template('about.html', page='about')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))

    app.run(
        host='0.0.0.0',
        port=port
    )