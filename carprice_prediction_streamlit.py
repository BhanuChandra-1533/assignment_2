import streamlit as st
import webbrowser
import pandas as pd
import plotly.express as px
from PIL import Image
import pickle
from sklearn.preprocessing import LabelEncoder

# Load datasets
df = pd.read_csv(r"C:\Users\bhanu\Downloads\dont_ordinalencoding_cardekho.csv")
df1 = pd.read_csv(r"C:\Users\bhanu\Downloads\ml_cardekho.csv")
df2 = pd.read_csv(r"C:\Users\bhanu\Downloads\cardekhoDATA.csv")

# Set page icon and layout
# icon = Image.open(r"C:\Users\ADMIN\Videos\career_fair\carDekho\download.png")
# st.set_page_config(page_title='CarDekho Used Car Price Prediction', page_icon=icon, layout="wide", initial_sidebar_state="auto")

# Sidebar Menu
with st.sidebar:
    st.markdown("# :rainbow[Select an option to filter:]")
    selected = st.selectbox("**Menu**", ("Home", "Analysis", "Prediction"))

# Helper function
def inverse(x):
    return 1 / x if x != 0 else None

# Label encoders for categorical data
encoders = {
    'fuel_type': LabelEncoder().fit(df['fuel_type']),
    'transmission_type': LabelEncoder().fit(df['transmission_type']),
    'brand': LabelEncoder().fit(df['brand']),
    'model': LabelEncoder().fit(df['model']),
    'location': LabelEncoder().fit(df['location'])
}

# Home Page
if selected == "Home":
    st.markdown('## :green[Welcome to Home Page:]')
    st.markdown('## :blue[Project Title:]')
    st.subheader("CarDekho Used Car Price Prediction")
    st.markdown('## :blue[Skills Acquired:]')
    st.subheader("Data Cleaning, EDA, Visualization, and Machine Learning")
    st.markdown('## :blue[Domain:]')
    st.subheader("Automobile")
    st.markdown('## :blue[Problem Statement:]')
    st.subheader("Create a model to predict used car prices accurately based on car features, age, mileage, fuel type, and more.")

    # Links (update with actual URLs)
    github_link = 'https://github.com/your_repo'
    linkedin_link = 'https://www.linkedin.com/in/your_profile'
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button('GitHub'):
            webbrowser.open_new_tab(github_link)
    with col2:
        if st.button('LinkedIn'):
            webbrowser.open_new_tab(linkedin_link)

# Prediction Page
if selected == 'Prediction':
    st.markdown('## :green[Welcome to Car Price Prediction:]')

    with st.form(key='form'):
        # User input fields
        fuel_type = st.radio("Select fuel type:", df['fuel_type'].unique())
        kilometer = st.number_input(f'Enter kilometers driven (Min: {df2["kilometer"].min()}, Max: {df2["kilometer"].max()})', format="%.1f")
        transmission_type = st.radio("Select transmission type:", df['transmission_type'].unique())
        owner_no = st.number_input("Enter number of owners", format="%.0f")
        brand = st.selectbox("Select a brand:", df['brand'].unique())
        model = st.selectbox("Select a model:", df['model'].unique())
        model_year = st.number_input("Enter manufacturing year", format="%.0f")
        seats_count = st.number_input(f'Enter seat count (Min: {df["seats_count"].min()}, Max: {df["seats_count"].max()})', format="%.0f")
        car_engine_cc = st.number_input(f'Enter engine capacity (cc) (Min: {df["car_engine_cc"].min()}, Max: {df["car_engine_cc"].max()})', format="%.0f")
        mileage = st.number_input(f'Enter mileage (Min: {df["mileage"].min()}, Max: {df["mileage"].max()})', format="%.1f")
        location = st.selectbox("Select location:", df['location'].unique())
        age = st.number_input("Enter age of the car", format="%.0f")

        # Convert categorical inputs to numerical
        fuel_type_enc = encoders['fuel_type'].transform([fuel_type])[0]
        transmission_type_enc = encoders['transmission_type'].transform([transmission_type])[0]
        brand_enc = encoders['brand'].transform([brand])[0]
        model_enc = encoders['model'].transform([model])[0]
        location_enc = encoders['location'].transform([location])[0]

        # Load model
        with open(r"C:\Users\bhanu\Downloads\random_regression.pkl", 'rb') as file:
            model = pickle.load(file)

        if st.form_submit_button('Submit'):
            # Prepare input features
            kilo_inverse = inverse(kilometer)
            features = [[fuel_type_enc, kilo_inverse, transmission_type_enc, owner_no, model_enc, model_year, seats_count, car_engine_cc, mileage, location_enc, age]]
            
            # Ensure features are converted to the correct dtype for prediction
            prediction = model.predict(features)
            st.write(f"# :green[Estimated Price: :red[{prediction[0]:,.2f}]]")

# Analysis Page
if selected == 'Analysis':
    # Line Plot: Total price by registration year
    st.markdown("### Total Price by Registration Year")
    data_yearly = df2.groupby('registration_year').sum()['price']
    fig = px.line(data_yearly, x=data_yearly.index, y='price', title="Total Price by Year")
    fig.update_traces(line_color='#8EF316', line={'width':3})
    st.plotly_chart(fig, use_container_width=True)

    # Area Plot: Highest seat count by brand
    st.markdown("### Highest Seat Count by Brand")
    data_seats = df2.groupby('brand')['seats_count'].max().reset_index()
    fig = px.area(data_seats, x='brand', y='seats_count', color='seats_count', title="Highest Seat Count by Brand")
    st.plotly_chart(fig, use_container_width=True)

    # Histogram: Price distribution by selected column and year
    selected_year = st.slider("Select year:", int(df2['registration_year'].min()), int(df2['registration_year'].max()))
    selected_column = st.selectbox("Select feature for analysis:", df2.columns.drop('price'))
    yearly_data = df2[df2['registration_year'] == selected_year].groupby(selected_column).sum()['price']
    fig = px.histogram(yearly_data, x=yearly_data.index, y='price', title=f"{selected_column} vs Price")
    st.plotly_chart(fig, use_container_width=True)

    # Additional analysis plots (if needed)
    # E.g., Scatter plot of kilometer vs price
    st.markdown("### Kilometers Driven vs Price")
    fig = px.scatter(df2, x='kilometer', y='price', size='price', color='price', log_x=True, size_max=50, title="Kilometers Driven vs Price")
    st.plotly_chart(fig, use_container_width=True)


     






    


        
   
        
   