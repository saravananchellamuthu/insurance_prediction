import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title = 'Insurance Premium Prediction App', page_icon = '🏥')

st.markdown("<h1 style='text-align: center; color: white;'>Health Insurance Premium Prediction App</h1>", unsafe_allow_html=True)

name = st.text_input('Enter your name:')

st.subheader(f'Hi {name}')

st.markdown(f'''The Insurance Premium Prediction App estimates potential medical expenses based on individual 
            attributes and health status, using a model trained on historical data from past insurance holders. 
            Users input their information to receive an estimate of medical expenses, enabling them 
            to focus on their health while understanding potential insurance policy charges''')

# inputs
sex = st.selectbox("What is your gender?", ('male', 'female'))

age = st.slider("What is your age?", min_value=18, max_value=64)

bmi = st.slider("What is your BMI?", min_value=16.0, max_value=53.1, help='Body Mass Index(BMI) = Weight/Height^2')

children = st.selectbox("How many children do you have?", ('0', '1', '2', '3', '4', '5'))

smoker = st.selectbox("Do you have smoking tendencies?", ('yes', 'no'))

region = st.selectbox("What region do you belong?", ('northwest', 'northeast', 'southwest', 'southeast'))

# col names and created df for inputs
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], columns = cols)


def feature_preprocessing(df, encoder, scaler):
    # Categorical columns
    cat_cols = ['sex', 'smoker', 'region']

    # Numerical columns
    num_cols = ['age', 'bmi', 'children']

    # Encode categorical columns
    cat_encoded = encoder.transform(df[cat_cols])
    cat_encoded_df = pd.DataFrame(
        cat_encoded, 
        index=df.index, 
        columns=encoder.get_feature_names_out(cat_cols))

    # Scale numerical columns
    num_scaled = scaler.transform(df[num_cols])
    num_scaled_df = pd.DataFrame(num_scaled, index=df.index, columns=num_cols)

    # Combine the encoded categorical and scaled numerical features
    processed_df = pd.concat([num_scaled_df, cat_encoded_df], axis=1)

    return processed_df

def predict_premium(input_data):
    # Load the encoder and scaler
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Preprocess the input data
    processed_data = feature_preprocessing(input_data, encoder, scaler)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict(processed_data)
    print(prediction)
    return prediction


if(st.button('Predict')):
    result = ''
    result = predict_premium(data)
    st.success('Your Estimated Medical Expenses Bill is {} USD'.format(np.around(*result, decimals=2)))
# remove main menu and footer note of streamlit
hide_menu_style = '''
       <style>
       #MainMenu {visibility: hidden;}
       footer{visibility: hidden;}
       </style>
        '''
st.markdown(hide_menu_style, unsafe_allow_html=True)