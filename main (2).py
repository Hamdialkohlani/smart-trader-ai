
import streamlit as st
import pyrebase
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import re
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
import ta
from sklearn.metrics import mean_squared_error
from firebase_config import firebase_config

# تهيئة Firebase
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

st.set_page_config(page_title="Smart Trader AI", layout="wide")
st.title("🔐 Smart Trader AI - دخول آمن")

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# تسجيل دخول تجريبي مباشر باستخدام البريد الذي زوّده المستخدم
email = "hamdialkohlany7769@gmail.com"
st.session_state['authenticated'] = True
st.session_state['user_email'] = email
st.success(f"✅ تم تسجيل الدخول تلقائيًا بالبريد: {email}")

if st.session_state['authenticated']:
    st.header("📈 لوحة التوقع الذكي")

    symbol = st.text_input("🔍 رمز السوق:", value="AAPL")
    period = st.selectbox("الفترة", ["7d", "1mo", "3mo", "6mo", "1y"])
    interval = st.selectbox("الفاصل الزمني", ["1h", "1d"])

    st.sidebar.header("⚙️ إعدادات النموذج")
    epochs = st.sidebar.slider("Epochs", 10, 200, 50)
    window_size = st.sidebar.slider("Window Size", 5, 50, 20)
    predict_len = st.sidebar.slider("عدد الشموع المستقبلية", 1, 20, 5)
    lstm_units = st.sidebar.slider("LSTM Units", 32, 256, 128)
    num_layers = st.sidebar.slider("عدد طبقات LSTM", 1, 3, 2)
    dropout_rate = st.sidebar.slider("Dropout", 0.0, 0.5, 0.2)

    @st.cache_data
    def load_data(symbol, period, interval):
        df = yf.download(symbol, period=period, interval=interval)
        df.dropna(inplace=True)
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['macd'] = ta.trend.MACD(df['Close']).macd()
        df['ema_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
        bb = ta.volatility.BollingerBands(close=df['Close'], window=20)
        df['bollinger_high'] = bb.bollinger_hband()
        df['bollinger_low'] = bb.bollinger_lband()
        df['volume'] = df['Volume']
        df.dropna(inplace=True)
        return df

    def create_lstm_model(input_shape):
        model = Sequential()
        for i in range(num_layers):
            return_seq = i < num_layers - 1
            model.add(LSTM(lstm_units, return_sequences=return_seq, input_shape=input_shape if i == 0 else None))
            model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_data(df, features, window_size):
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df[features])
        X, y = [], []
        for i in range(window_size, len(df_scaled) - predict_len):
            X.append(df_scaled[i - window_size:i])
            y.append(df_scaled[i:i+predict_len, 0])
        return np.array(X), np.array(y), scaler

    if st.button("ابدأ التوقع"):
        try:
            df = load_data(symbol, period, interval)
            st.success("✅ البيانات جاهزة للتحليل")
            features = ['Close', 'rsi', 'macd', 'ema_20', 'bollinger_high', 'bollinger_low', 'volume']
            X, y, scaler = prepare_data(df, features, window_size)
            model = create_lstm_model((X.shape[1], X.shape[2]))
            model.fit(X, y[:, 0], epochs=epochs, verbose=0)
            last_seq = df[features].values[-window_size:]
            last_scaled = scaler.transform(last_seq).reshape(1, window_size, len(features))
            preds = []
            input_seq = last_scaled
            for _ in range(predict_len):
                next_pred = model.predict(input_seq)[0][0]
                preds.append(next_pred)
                input_seq = np.append(input_seq[:, 1:, :], [[[next_pred]*len(features)]], axis=1)
            preds_rescaled = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
            for i, val in enumerate(preds_rescaled, 1):
                st.write(f"📊 الشمعة {i}: {val:.2f}")
        except Exception as e:
            st.error(f"❌ حدث خطأ أثناء التوقع: {e}")
else:
    st.warning("🔒 الرجاء تسجيل الدخول لاستخدام لوحة الذكاء الاصطناعي.")
