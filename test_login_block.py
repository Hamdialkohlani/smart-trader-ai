
method = st.radio("اختر طريقة الدخول:", ["📧 بريد إلكتروني", "🔐 حساب Google", "🧬 بصمة الجهاز"])

if method == "📧 بريد إلكتروني":
    email = st.text_input("📧 بريدك الإلكتروني")
    if st.button("✉️ إرسال كود تحقق"):
        st.session_state['authenticated'] = True
        st.session_state['user_email'] = email
        st.success("✅ تم تسجيل الدخول مؤقتًا للاختبار.")
elif method == "🔐 حساب Google":
    st.warning("🔗 تسجيل الدخول عبر Google يتطلب تفعيل OAuth من Firebase Console.")
elif method == "🧬 بصمة الجهاز":
    st.info("💡 المصادقة البيومترية تعمل فقط في تطبيقات الويب المتقدمة.")
