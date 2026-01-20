import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the saved pipeline
pipeline = joblib.load("svm_sentiment_model.pkl")

# --- Page Config ---
st.set_page_config(
    page_title="Flipkart Sentiment Dashboard",
    page_icon="🛒",
    layout="wide"
)

# --- Custom CSS ---
st.markdown(
    """
    <style>
    /* Force entire app background to white */
    .stApp {
        background-color: #FFFFFF;
    }

    /* Force all text to black */
    .stApp, .stMarkdown, .stTextInput label, .stTextArea label, .stNumberInput label {
        color: #000000 !important;
    }

    /* Input box text color */
    input, textarea {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }

    /* Navbar */
    .navbar {
        overflow: hidden;
        background-color: #2874F0;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .navbar a {
        float: left;
        display: block;
        color: white;
        text-align: center;
        padding: 12px 16px;
        text-decoration: none;
        font-size: 18px;
        font-weight: bold;
    }
    .navbar a:hover {
        background-color: #1A4FB3;
        border-radius: 5px;
    }

    /* Footer */
    .footer {
        background-color: #2874F0;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-size: 14px;
        margin-top: 30px;
    }

    /* Prediction box */
    .prediction-box {
        border-radius: 10px;
        padding: 15px;
        font-size: 20px;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)
       

# --- Navbar ---
st.markdown(
    """
    <div class="navbar">
        <a href="#">🏠 Home</a>
        <a href="#">📞 Contact</a>
        <a href="#">❓ Help</a>
        <a href="#">ℹ️ More Details</a>
    </div>
    """,
    unsafe_allow_html=True
)
# Navbar simulation
menu = st.radio("Navigation", ["Home", "Contact", "Help", "More Details"], horizontal=True)

if menu == "Home":
    st.header("🏠 Home")
    st.write("This is the main sentiment analysis dashboard.")
    # put your review input + prediction code here

elif menu == "Contact":
    st.header("📞 Contact")
    st.write("For queries, reach out to: support@flipkart.com")

elif menu == "Help":
    st.header("❓ Help")
    st.write("Enter a product review and click Predict Sentiment to see results.")

elif menu == "More Details":
    st.header("ℹ️ More Details")
    st.write("This app uses a Linear SVM model trained on Flipkart reviews.")


# --- Header ---
st.markdown("<h1 style='text-align:center; color:#2874F0;'>🛍️ Flipkart Review Sentiment Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>✨ 🖊️ 📦 💬 🎨 🛒 ✨</p>", unsafe_allow_html=True)

# --- User Input ---
st.subheader("🔎 Enter Review Details")
review_text = st.text_area("Review Text", placeholder="Type the customer review here...")
summary_text = st.text_input("Summary (optional)", "")
price = st.number_input("Product Price", min_value=0, value=0)
product_name = st.text_input("Product Name", "demo_product")

# --- Prediction ---
if st.button("Predict Sentiment"):
    input_df = pd.DataFrame({
        "Review": [review_text],
        "Summary": [summary_text],
        "product_price": [price],
        "product_name": [product_name]
    })
    
    prediction = pipeline.predict(input_df)[0]
    
    # --- Fancy Output ---
    if prediction == "positive":
        st.markdown("<div class='prediction-box' style='background-color:#D4EDDA; color:#155724;'>🌟 Sentiment: Positive</div>", unsafe_allow_html=True)
    elif prediction == "negative":
        st.markdown("<div class='prediction-box' style='background-color:#F8D7DA; color:#721C24;'>⚠️ Sentiment: Negative</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='prediction-box' style='background-color:#FFF3CD; color:#856404;'>😐 Sentiment: Neutral</div>", unsafe_allow_html=True)

    # --- Graphs Section ---
    st.subheader("📈 Sentiment Distribution (Demo)")
    sentiments = ["Positive", "Negative", "Neutral"]
    values = [40, 25, 35]  # Example distribution, replace with real stats
    
    col1, col2 = st.columns(2)
    
    # Bar Chart
    with col1:
        fig, ax = plt.subplots()
        ax.bar(sentiments, values, color=["#28a745", "#dc3545", "#ffc107"])
        ax.set_title("Sentiment Distribution (Bar Chart)")
        st.pyplot(fig)
    
    # Pie Chart
    with col2:
        fig2, ax2 = plt.subplots()
        ax2.pie(values, labels=sentiments, autopct='%1.1f%%', colors=["#28a745", "#dc3545", "#ffc107"])
        ax2.set_title("Sentiment Distribution (Pie Chart)")
        st.pyplot(fig2)

# --- Footer ---
st.markdown("<div class='footer'>Made with ❤️ by Tanushri | Powered by Linear SVM</div>", unsafe_allow_html=True)