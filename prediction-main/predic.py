import pandas as pd
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title=" Prediction App", page_icon="🔮", layout="centered")

st.title('Prediction App')

data = pd.read_csv("prediction-main/player_rankings_2024.csv")
data['Value'] = data['Value'].str.replace('$', '', regex=False)
data['Value'] = data['Value'].str.replace(',', '', regex=False)
data['Value'] = data['Value'].astype(int)

# Prepare the training and testing data
X = data.drop(['Rank', 'Player', 'Team', 'Salary', 'Value'], axis=1)
y = data['Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
regr = LinearRegression()
regr.fit(X_train, y_train)


st.write("Enter the player's statistics to predict their Value:")

# Input boxes for RAA, Wins, and EFscore
RAA = st.number_input('RAA', value=0)
Wins = st.number_input('Wins', value=0.0 , format="%.3f")
EFscore = st.number_input('EFscore', value=0.0 , format="%.3f")


if st.button('Predict Value'):
    new_data = pd.DataFrame({
        'RAA': [RAA],       
        'Wins': [Wins],     
        'EFscore': [EFscore]    
    })
    new_predictions = regr.predict(new_data)
    st.write(f"Predicted Value: ${new_predictions[0]:,.2f}")

