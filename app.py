import streamlit as st
import pandas as pd

st.title("Diet Meal Planning Optimisation using Evolutionary Programming")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Food_and_Nutrition__.csv")

data = load_data()

st.subheader("Food and Nutrition Dataset")
st.dataframe(data)

