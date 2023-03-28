import joblib
import streamlit as st

sentiment = joblib.load('C:/Users/Admin 21/Downloads/STREAMLIT/model.pkl')
vectorizer = joblib.load('C:/Users/Admin 21/Downloads/STREAMLIT/vector.pkl')

def main(title = " Tweet Sentiments Classification Streamlit App"):
    
    with st.expander("1. Check if a tweet has a negative or positive sentiment"):
        tweet = st.text_input("Enter the tweet")
        if st.button("Predict"):
            prediction = sentiment.predict(vectorizer.transform([tweet]))
            
            if prediction == 1:
                info = "Positive sentiment"
            else:
                info ="Negative sentiment"
            
            st.success('Prediction: {}'.format(info))
        
if __name__ == "__main__":
    main()