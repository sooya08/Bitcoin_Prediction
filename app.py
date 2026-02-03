from flask import Flask, render_template, jsonify, request
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import utils

app = Flask(__name__)

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
        # Price Trend Plot
        fig_price = px.line(data, x='date', y='close', title="Bitcoin Price Trend (USD)")
        fig_price.update_layout(
            template="plotly_dark", 
            font=dict(family="Inter, sans-serif"),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        plots['price'] = json.dumps(fig_price, cls=plotly.utils.PlotlyJSONEncoder)

        # Volume Plot (Area Chart)
        fig_volume = px.area(data, x='date', y='volume', title="Bitcoin Trading Volume Over Time")
        fig_volume.update_layout(
            template="plotly_dark", 
            font=dict(family="Inter, sans-serif"),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        plots['volume'] = json.dumps(fig_volume, cls=plotly.utils.PlotlyJSONEncoder)

        # Statistics
        stats = data.describe().to_html(classes="table table-dark table-striped", float_format="%.2f")
        
        # Dataset Preview
        cols = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        available_cols = [c for c in cols if c in data.columns]
        dataset_preview = data[available_cols].tail(50).to_html(classes="table table-dark table-striped table-hover", float_format="%.2f", index=False)
        
        # Seasonality Analysis (Monthly Returns)
        seasonality = utils.get_seasonality_data()
    else:
        stats = None
        seasonality = None

    return render_template('eda.html', page='eda', plots=plots, stats=stats, dataset_preview=dataset_preview, seasonality=seasonality)

@app.route('/live')
def live_price():
    return render_template('live_price.html', page='live')

@app.route('/api/live_price')
def get_live_price_api():
    usd, inr = utils.get_live_price()
    return jsonify({'usd': usd, 'inr': inr})

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    result = None
    if request.method == 'POST':
        # Trigger training
        result = utils.train_xgb_model()
        
        if result and 'error' not in result:
            # Create Plotly figure for test vs predicted
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=result['y_test'], mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(y=result['predictions'], mode='lines', name='Predicted'))
            fig.update_layout(title="Actual vs Predicted (Test Data)", template="plotly_dark", font=dict(family="Inter, sans-serif"))
            result['plot'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
    return render_template('analysis.html', page='analysis', result=result)

@app.route('/prediction', methods=['GET'])
def prediction():
    days = int(request.args.get('days', 30))
    prediction_data = utils.predict_next_days(days)
    
    plot = None
    metrics = None
    if prediction_data:
        # Create CoinCodex Style Graph: History + Forecast
        fig = go.Figure()
        
        # Trace 1: Historical Data (last 30 days)
        fig.add_trace(go.Scatter(
            x=prediction_data['history_dates'], 
            y=prediction_data['history_prices'],
            mode='lines',
            name='Historical',
            line=dict(color='#888888', width=2)
        ))
        
        # Trace 2: Predicted Data (connected to history)
        # We add the last historical point to the start of prediction for a smooth line
        pred_x = [prediction_data['history_dates'][-1]] + prediction_data['dates']
        pred_y = [prediction_data['history_prices'][-1]] + prediction_data['prices']
        
        fig.add_trace(go.Scatter(
            x=pred_x, 
            y=pred_y,
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#f7931a', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title="Bitcoin Price Prediction: History & Forecast",
            template="plotly_dark",
            hovermode="x unified",
            font=dict(family="Inter, sans-serif"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        plot = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        plot_obj = json.loads(plot) # Pass as dict for table iteration
        
        metrics = {
            "current_price": prediction_data['current_price'],
            "next_day_price": prediction_data['next_day_price'],
            "price_diff": prediction_data['price_diff'],
            "diff_percent": prediction_data['diff_percent']
        }
        
    return render_template('prediction.html', page='prediction', plot=plot, plot_obj=plot_obj, days=days, metrics=metrics)

@app.route('/about')
def about():
    return render_template('about.html', page='about')

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
