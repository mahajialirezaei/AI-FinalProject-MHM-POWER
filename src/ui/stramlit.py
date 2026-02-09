import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

# --- تنظیمات صفحه ---
st.set_page_config(page_title="Bank Marketing Predictor", layout="wide")

# --- بارگذاری مدل‌ها و پیش‌پردازشگر (Cache شده) ---
@st.cache_resource
def load_assets():
    # بارگذاری پیش‌پردازشگر (ColumnTransformer)
    with open('./../preprocessing/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    # بارگذاری مدل‌ها
    model_names = ['xgboost_optimized', 'xgboost_weighted_optimized', 'baseline_logreg', 'xgboost_weighted']
    models = {}
    for name in model_names:
        try:
            with open(f'./../models/{name}.pkl', 'rb') as f:
                models[name] = pickle.load(f)
        except:
            st.warning(f"فایل {name}.pkl یافت نشد.")
    
    return preprocessor, models

preprocessor, models_dict = load_assets()

# --- عنوان برنامه ---
st.title("🏦 سامانه تحلیل هوشمند بازاریابی بانکی")
st.markdown("این مدل بر اساس ویژگی‌های مشتری، احتمال خرید سپرده مدت‌دار را پیش‌بینی می‌کند.")

# --- طراحی بدنه ورودی‌ها (دقیقاً مطابق لیست شما) ---
with st.sidebar:
    st.header("📋 مشخصات مشتری")
    
    # ویژگی‌های عددی
    age = st.number_input("سن (Age)", 18, 95, 35)
    balance = st.number_input("موجودی حساب (Balance)", -8000, 100000, 2000)
    day = st.slider("روز آخرین تماس (Day)", 1, 31, 15)
    campaign = st.number_input("تعداد تماس در این کمپین", 1, 50, 1)
    pdays = st.number_input("روزهای گذشته از تماس قبلی (Pdays)", -1, 999, -1)
    previous = st.number_input("تعداد تماس‌های قبلی (Previous)", 0, 50, 0)

    # ویژگی‌های دسته‌ای
    job = st.selectbox("شغل", ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"])
    marital = st.selectbox("وضعیت تاهل", ["married","divorced","single"])
    education = st.selectbox("تحصیلات", ["unknown","secondary","primary","tertiary"])
    default = st.selectbox("بدهی معوقه؟", ["no", "yes"])
    housing = st.selectbox("وام مسکن؟", ["no", "yes"])
    loan = st.selectbox("وام شخصی؟", ["no", "yes"])
    contact = st.selectbox("نوع تماس", ["cellular","telephone","unknown"])
    month = st.selectbox("ماه", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    poutcome = st.selectbox("نتیجه کمپین قبلی", ["unknown","other","failure","success"])

# --- ساخت دیتافریم از ورودی کاربر ---
# نکته: ترتیب ستون‌ها مهم نیست چون ColumnTransformer بر اساس نام کار می‌کند
user_data = pd.DataFrame([{
    'age': age, 'job': job, 'marital': marital, 'education': education, 
    'default': default, 'balance': balance, 'housing': housing, 'loan': loan, 
    'contact': contact, 'day': day, 'month': month, 'campaign': campaign, 
    'pdays': pdays, 'previous': previous, 'poutcome': poutcome
}])

# جایگزینی unknown با NaN مشابه کد شما
user_data.replace('unknown', np.nan, inplace=True)

# --- دکمه اجرا و تحلیل ---
if st.button("🚀 تحلیل توسط مدل‌های هوش مصنوعی"):
    
    # ۱. اعمال پیش‌پردازش (فرایند مشابه کد آموزشی شما)
    # ColumnTransformer به صورت خودکار ایمپوتر و اسکیلر را اعمال می‌کند
    try:
        processed_input = preprocessor.transform(user_data)
        
        # ۲. گرفتن پیش‌بینی از تمامی مدل‌ها
        results = []
        for name, model in models_dict.items():
            # استفاده از [0][1] برای گرفتن احتمال کلاس مثبت (Yes)
            prob = model.predict_proba(processed_input)[0][1]
            results.append({"مدل": name, "احتمال موفقیت": prob})
        
        # ۳. نمایش نتایج
        res_df = pd.DataFrame(results).sort_values(by="احتمال موفقیت", ascending=False)
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.write("### 🏆 رتبه‌بندی مدل‌ها")
            st.dataframe(res_df.style.format({"احتمال موفقیت": "{:.2%}"})
                         .background_gradient(cmap='RdYlGn'), use_container_width=True)
            
        with col2:
            st.write("### 📊 مقایسه احتمال‌ها")
            fig = px.bar(res_df, x='احتمال موفقیت', y='مدل', orientation='h',
                         color='احتمال موفقیت', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
            
        # ۴. تحلیل نهایی
        avg_prob = res_df["احتمال موفقیت"].mean()
        st.divider()
        if avg_prob > 0.5:
            st.success(f"✅ تحلیل کلی: با میانگین احتمال {avg_prob:.1%}, پیشنهاد می‌شود با این مشتری تماس گرفته شود.")
        else:
            st.warning(f"❌ تحلیل کلی: احتمال موفقیت پایین است ({avg_prob:.1%}).")

    except Exception as e:
        st.error(f"خطا در پردازش داده‌ها: {e} - مطمئن شوید که preprocessor.pkl با نسخه فعلی scikit-learn سازگار است.")