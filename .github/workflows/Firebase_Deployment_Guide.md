
# 🚀 دليل نشر موقع GulfScope على Firebase Hosting

هذا الدليل يوضح خطوات نشر موقعك مجانًا باستخدام Firebase Hosting، بسرعة وسهولة.

---

## ✅ المتطلبات:
- تثبيت Node.js
- إنشاء مشروع في Firebase (تم ✅)
- اتصال بالإنترنت
- امتلاك ملفات المشروع (HTML, CSS, JS)

---

## 🧭 خطوات النشر:

### 1️⃣ تثبيت أدوات Firebase:
```bash
npm install -g firebase-tools
```

---

### 2️⃣ تسجيل الدخول إلى Firebase:
```bash
firebase login
```

---

### 3️⃣ إعداد المجلد:
```bash
mkdir gulfscope
cd gulfscope
```
ثم انقل ملفات المشروع إلى هذا المجلد.

---

### 4️⃣ تهيئة Firebase:
```bash
firebase init
```
- اختر: Hosting
- اختر مشروعك من Firebase
- اسم مجلد النشر: `public`
- وافق على استخدام `index.html` كنقطة دخول
- اختر "لا" عند السؤال عن SPA

---

### 5️⃣ النشر:
```bash
firebase deploy
```

---

## 🌐 رابط موقعك:
سيظهر رابط مثل:
```
Hosting URL: https://gulfscope-77708.web.app
```

---

## 📌 ملاحظات:
- لتحديث الموقع: عدّل الملفات ثم أعد `firebase deploy`
- الربط مع Realtime Database يعمل تلقائيًا
- لا حاجة لخوادم خارجية

---

© 2025 GulfScope - جميع الحقوق محفوظة
