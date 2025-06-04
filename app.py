import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import warnings
import requests
from streamlit_lottie import st_lottie
import os



warnings.filterwarnings("ignore")
st.set_page_config(page_title="Real-Time Stock Predictor", layout="wide",page_icon="📈")

PREDICTION_DAYS = 30
TIME_STEP = 60
DATA_YEARS = 3

model = load_model('stock_price_model.h5')
model.make_predict_function()  

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_stock = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_1pxqjqps.json")
lottie_loading = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_usmfx6bp.json")

st.markdown(
    """
    <style>
    body, .main {
        background: linear-gradient(135deg, #232526 0%, #414345 100%) !important;
        color: #fff;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        font-weight: bold;
        color: #00b894;
    }
    .stTextInput>div>div>input {
        background: #222;
        color: #fff;
        border-radius: 8px;
        border: 1px solid #00b894;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00b894 0%, #00cec9 100%);
        color: #fff;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #00cec9 0%, #00b894 100%);
        color: #222;
    }
    .card {
        background: rgba(34, 49, 63, 0.85);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 24px 0 rgba(0,0,0,0.2);
    }
    .price-up {color: #00ff99; font-weight: bold; font-size: 1.5em;}
    .price-down {color: #ff7675; font-weight: bold; font-size: 1.5em;}
    </style>
    """,
    unsafe_allow_html=True
)

def preprocess_data(df):
    """Process yfinance data"""
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.reset_index().rename(columns={'index': 'Date'})
    df = df[['Date', 'High', 'Low', 'Open', 'Close', 'Volume']]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def get_stock_data(stock_symbol):
    """Fetch stock data with caching"""
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=365 * DATA_YEARS)
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    return preprocess_data(df)


def prepare_data(df):
    """Prepare data for LSTM prediction"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    X = np.array([scaled_data[i:i + TIME_STEP, 0]
                  for i in range(len(scaled_data) - TIME_STEP - 1)])
    y = scaled_data[TIME_STEP + 1:, 0]

    return X.reshape(X.shape[0], TIME_STEP, 1), y, scaler


def predict_future(model, data, scaler):
    """Generate future predictions"""
    last_data = data[-TIME_STEP:].reshape(1, TIME_STEP, 1)
    future_preds = np.zeros(PREDICTION_DAYS, dtype='float32')

    for i in range(PREDICTION_DAYS):
        next_pred = model.predict(last_data, verbose=0)[0, 0]
        future_preds[i] = next_pred
        last_data = np.roll(last_data, -1, axis=1)
        last_data[0, -1, 0] = next_pred

    return scaler.inverse_transform(future_preds.reshape(-1, 1))


def create_plot(df, pred_data=None, future_data=None, title=""):
    """Create interactive Plotly figure"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        name='Actual Price',
        line=dict(color='#00b894')
    ))

    if pred_data is not None:
        fig.add_trace(go.Scatter(
            x=df.index[TIME_STEP + 1:],
            y=pred_data[:, 0],
            name='Predicted',
            line=dict(color='#fdcb6e')
        ))
    if future_data is not None:
        future_dates = pd.date_range(
            start=df.index[-1],
            periods=PREDICTION_DAYS + 1
        )[1:]
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_data[:, 0],
            name='30-Day Forecast',
            line=dict(color='#0984e3', dash='dash')
        ))

    fig.update_layout(
        title=title,
        template='plotly_dark',
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family='Segoe UI, Roboto, sans-serif', color='#fff'),
        plot_bgcolor='rgba(34,49,63,0.95)',
        paper_bgcolor='rgba(34,49,63,0.95)'
    )
    return fig


def main():
    st_lottie(lottie_stock, height=120, key="header_anim")
    st.markdown("""
        <h1 style='text-align: center; color: #00b894; font-size: 3em; font-weight: bold;'>Real-Time Stock Predictor</h1>
        <p style='text-align: center; color: #fff; font-size: 1.2em;'>Predict stock prices using LSTM neural networks</p>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        stock_symbol = st.text_input(
            "Stock Symbol (Examples: TSLA, AAPL, MSFT, AMZN, GOOG, AEP)",
            value="TSLA"
        )
        predict_btn = st.button("Predict", use_container_width=True)
    
    if predict_btn and stock_symbol:
        with st.spinner('Predicting...'):
            st_lottie(lottie_loading, height=80, key="loading_anim")
            try:
                df = get_stock_data(stock_symbol)
                X, y, scaler = prepare_data(df)
                y_pred = model.predict(X)
                y_pred = scaler.inverse_transform(y_pred)
                future_prices = predict_future(
                    model,
                    scaler.transform(df['Close'].values.reshape(-1, 1)),
                    scaler
                )
                last_price = df['Close'].iloc[-1]
                last_date = df.index[-1].strftime('%Y-%m-%d')
                prev_price = df['Close'].iloc[-2] if len(df) > 1 else last_price
                price_class = "price-up" if last_price >= prev_price else "price-down"
                price_icon = "🔺" if last_price >= prev_price else "🔻"
                st.markdown(f"""
                    <div class='card' style='text-align:center;'>
                        <span class='{price_class}'>{price_icon} ${last_price:.2f}</span><br>
                        <span style='color:#b2bec3;'>Last Date: {last_date}</span>
                    </div>
                """, unsafe_allow_html=True)
                tab1, tab2, tab3 = st.tabs([
                    "📈 Price Prediction", "⏳ 30-Day Forecast", "📊 Technical Indicators"])
                with tab1:
                    st.plotly_chart(create_plot(df, pred_data=y_pred, title=f"{stock_symbol} Price Prediction"), use_container_width=True)
                with tab2:
                    st.plotly_chart(create_plot(df, future_data=future_prices, title=f"{stock_symbol} 30-Day Forecast"), use_container_width=True)
                with tab3:
                    df['SMA_50'] = df['Close'].rolling(50).mean()
                    df['SMA_200'] = df['Close'].rolling(200).mean()
                    tech_fig = go.Figure()
                    tech_fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#00b894')))
                    tech_fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='50-Day SMA', line=dict(color='#fdcb6e')))
                    tech_fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='200-Day SMA', line=dict(color='#d63031')))
                    tech_fig.update_layout(title=f"{stock_symbol} Technical Indicators", template='plotly_dark',
                        font=dict(family='Segoe UI, Roboto, sans-serif', color='#fff'),
                        plot_bgcolor='rgba(34,49,63,0.95)',
                        paper_bgcolor='rgba(34,49,63,0.95)')
                    st.plotly_chart(tech_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    else:
        pass


if __name__ == "__main__":
    main()