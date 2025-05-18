# 📊 Smart Trader AI

تطبيق ذكاء اصطناعي تفاعلي يعمل على توقع اتجاه السوق بناءً على بيانات حقيقية من Yahoo Finance، ويستخدم نماذج LSTM المتقدمة في التعلم العميق.

## ✅ المميزات

- تسجيل دخول آمن (بريد إلكتروني / Google / بصمة)
- توقع اتجاه 5 شموع قادمة باستخدام LSTM
- تحليل فني باستخدام مؤشرات RSI، MACD، EMA، Bollinger Bands
- تجربة سهلة وسريعة باستخدام Streamlit
- واجهة مبنية بلغة Python وتدعم العربية بالكامل

## ⚙️ التقنية المستخدمة

- Streamlit
- TensorFlow / Keras
- scikit-learn
- yfinance
- ta (Technical Analysis Indicators)
- pyrebase4
- Firebase Authentication

## 🛠️ طريقة التشغيل محليًا

```bash
git clone https://github.com/YOUR_USERNAME/smart-trader-ai.git
cd smart-trader-ai
pip install -r requirements.txt
streamlit run main.py
```

## 🔒 إعدادات Firebase

يجب إضافة ملف `firebase_config.py` في نفس المجلد، يحتوي على:

```python
firebase_config = {
    "apiKey": "xxx",
    "authDomain": "xxx.firebaseapp.com",
    "projectId": "xxx",
    "storageBucket": "xxx.appspot.com",
    "messagingSenderId": "xxx",
    "appId": "xxx",
    "databaseURL": ""
}
```

## 💡 من إنشاء وتطوير:

**Hamdi Alkohlany**  
📧 hamdialkohlany7769@gmail.com

---
> هذا المشروع تجريبي ويهدف إلى بناء أدوات تداول ذكية تحترم ثقة المستخدم وتستخدم بيانات واقعية فقط.
