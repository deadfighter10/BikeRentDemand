import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # Keep for potential re-use if needed
from sklearn.ensemble import GradientBoostingRegressor  # Keep model import
from sklearn.preprocessing import StandardScaler  # Keep scaler import
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error  # Keep metrics
import joblib
import os

# --- Set Page Config FIRST ---
st.set_page_config(layout="wide", page_title="Bike Demand Predictor")

# --- Configuration & Constants ---
DATA_FILE_PATH = "Bike Sharing DC Dataset/day.csv"
# Using V2 model/scaler trained with de-normalized data and without 'yr'
MODEL_FILE_PATH = "gradient_boosting_model_v2.joblib"
SCALER_FILE_PATH = "scaler_v2.joblib"
FEATURE_NAMES_FILE = "feature_names_v2.joblib"  # File storing feature order
# Original features dropped during V2 training
FEATURES_TO_DROP_TRAINING = ["cnt", "instant", "dteday", "casual", "registered", "yr", 'temp', 'atemp', 'hum',
                             'windspeed']  # Include original normalized weather cols
# Define original scale calculation constants
T_MIN, T_MAX = -8, 39
AT_MIN, AT_MAX = -16, 50
HUM_MAX = 100
WIND_MAX = 67


# --- Helper Functions ---

def derive_season(month):
    """Derives season code (1-4) from month (1-12)."""
    if month in [12, 1, 2]:
        return 4  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    elif month in [9, 10, 11]:
        return 3  # Fall
    else:
        return 1  # Default or error case


def derive_workingday(weekday, holiday):
    """Derives workingday status (0 or 1) from weekday (0-6) and holiday (0/1)."""
    if holiday == 1:
        return 0  # Holiday is not a working day
    elif weekday in [0, 6]:  # 0=Sunday, 6=Saturday
        return 0  # Weekend is not a working day
    else:  # Weekdays 1-5 and not a holiday
        return 1  # It's a working day


def denormalize_data(df):
    """Converts normalized weather features back to original scales."""
    # This function might not be strictly needed anymore if load_and_prepare_data handles it
    # But keep it for potential standalone use or clarity
    df_copy = df.copy()
    df_copy['temp_c'] = df_copy['temp'] * (T_MAX - T_MIN) + T_MIN
    df_copy['atemp_c'] = df_copy['atemp'] * (AT_MAX - AT_MIN) + AT_MIN
    df_copy['hum_pct'] = df_copy['hum'] * HUM_MAX
    df_copy['windspeed_kmh'] = df_copy['windspeed'] * WIND_MAX
    df_copy = df_copy.drop(columns=['temp', 'atemp', 'hum', 'windspeed'], errors='ignore')
    return df_copy


# --- Caching Functions (Load Data, Model, Scaler, Feature Names) ---

@st.cache_data
def load_and_prepare_data(file_path):
    """Loads, de-normalizes, and prepares the bike sharing data."""
    if not os.path.exists(file_path):
        st.error(f"Error: Data file not found at {file_path}")
        st.stop()
    try:
        data = pd.read_csv(file_path)
        data_processed = denormalize_data(data)
        if 'cnt' not in data_processed.columns:
            st.error("Error: 'cnt' column missing after processing.")
            st.stop()
        return data_processed
    except Exception as e:
        st.error(f"Error loading/processing data: {e}")
        st.stop()


@st.cache_resource
def load_prediction_assets(model_path, scaler_path, features_path):
    """Loads the trained model, scaler, and feature names."""
    if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
        st.error(f"Error: Model/Scaler/Feature file(s) not found. "
                 f"Please ensure '{model_path}', '{scaler_path}', and '{features_path}' exist. "
                 "You may need to run the training script first.")
        st.stop()
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(features_path)
        print("Loaded V2 model, scaler, and feature names.")
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading prediction assets: {e}")
        st.stop()


# --- Load Assets ---
data_processed_df = load_and_prepare_data(DATA_FILE_PATH)
trained_model, fitted_scaler, FEATURE_ORDER = load_prediction_assets(
    MODEL_FILE_PATH, SCALER_FILE_PATH, FEATURE_NAMES_FILE
)

# --- Streamlit App Layout ---

st.title("🚲 Daily Bike Rental Demand Predictor")
st.markdown("""
Predict the **total number of daily bike rentals** based on date and weather conditions.
Season and working day status are automatically determined from your inputs.
*(Model: Gradient Boosting Regressor)*
""")

# --- Define Input Mappings ---
month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
             11: 'Nov', 12: 'Dec'}
weekday_map = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
weather_map = {1: "Clear / Few Clouds", 2: "Mist / Cloudy", 3: "Light Snow / Light Rain",
               4: "Heavy Rain / Thunderstorm / Snow"}
season_map_display = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}  # Only for display

# --- Define Tabs ---
tab1, tab2 = st.tabs(["📊 Make Prediction", "Explore Data"])

# --- Prediction Tab ---
with tab1:
    st.header("Input Conditions for Prediction")

    col1, col2, col3 = st.columns(3)  # Create 3 columns for inputs

    # Column 1: Date related inputs
    with col1:
        st.subheader("Date & Day Type")
        inp_mnth = st.selectbox("Month", options=list(month_map.keys()), format_func=lambda x: month_map[x],
                                key="pred_month", help="Select the month.")
        inp_weekday = st.selectbox("Day of Week", options=list(weekday_map.keys()),
                                   format_func=lambda x: weekday_map[x], key="pred_weekday", help="Select the day.")
        inp_holiday = st.selectbox("Holiday?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                                   key="pred_holiday", help="Is it a public holiday?")

    # Column 2: Weather Situation
    with col2:
        st.subheader("Weather Overview")
        weather_options = sorted(data_processed_df['weathersit'].unique())
        inp_weathersit = st.selectbox("Weather Situation", options=weather_options,
                                      format_func=lambda x: weather_map.get(x, f"Unknown: {x}"), key="pred_weather",
                                      help="Select the prevailing weather category for the day.")
        # Display derived season and working day for user confirmation
        derived_season = derive_season(inp_mnth)
        derived_workingday = derive_workingday(inp_weekday, inp_holiday)
        st.write(f"**Derived Season:** {season_map_display[derived_season]}")
        st.write(f"**Derived Working Day:** {'Yes' if derived_workingday == 1 else 'No'}")

    # Column 3: Weather Details (Typed Inputs)
    with col3:
        st.subheader("Weather Details")
        # Get descriptive stats for de-normalized data
        desc_stats = data_processed_df[FEATURE_ORDER].describe()

        inp_temp_c = st.number_input(
            "Temperature (°C)",
            min_value=round(desc_stats.loc['min', 'temp_c'] - 5, 1),  # Add some buffer
            max_value=round(desc_stats.loc['max', 'temp_c'] + 5, 1),
            value=round(desc_stats.loc['mean', 'temp_c'], 1),
            step=0.5, format="%.1f", key="pred_temp",
            help="Average daily temperature in Celsius."
        )
        inp_atemp_c = st.number_input(
            "Feeling Temperature (°C)",
            min_value=round(desc_stats.loc['min', 'atemp_c'] - 5, 1),
            max_value=round(desc_stats.loc['max', 'atemp_c'] + 5, 1),
            value=round(desc_stats.loc['mean', 'atemp_c'], 1),
            step=0.5, format="%.1f", key="pred_atemp",
            help="'Feels like' temperature in Celsius."
        )
        inp_hum_pct = st.number_input(
            "Humidity (%)",
            min_value=0.0,  # Humidity min is 0
            max_value=100.0,
            value=round(desc_stats.loc['mean', 'hum_pct'], 1),
            step=0.5, format="%.1f", key="pred_hum",
            help="Average daily humidity in percent."
        )
        inp_windspeed_kmh = st.number_input(
            "Windspeed (km/h)",
            min_value=0.0,  # Windspeed min is 0
            max_value=round(desc_stats.loc['max', 'windspeed_kmh'] + 10, 1),  # Add buffer
            value=round(desc_stats.loc['mean', 'windspeed_kmh'], 1),
            step=0.1, format="%.1f", key="pred_wind",
            help="Average daily windspeed in km/h."
        )

    st.markdown("---")  # Separator

    # --- Prediction Execution and Display ---
    predict_col, result_col = st.columns([0.3, 0.7])

    with predict_col:
        predict_button = st.button("🚀 Predict Demand", key="predict_button", type="primary", use_container_width=True)

    with result_col:
        if predict_button:
            # --- Derive features ---
            derived_season_val = derive_season(inp_mnth)
            derived_workingday_val = derive_workingday(inp_weekday, inp_holiday)

            # --- Create input DataFrame IN THE CORRECT ORDER ---
            input_dict = {
                'mnth': inp_mnth,
                'holiday': inp_holiday,
                'weekday': inp_weekday,
                'weathersit': inp_weathersit,
                'temp_c': inp_temp_c,
                'atemp_c': inp_atemp_c,
                'hum_pct': inp_hum_pct,
                'windspeed_kmh': inp_windspeed_kmh,
                # Add derived values
                'season': derived_season_val,
                'workingday': derived_workingday_val
            }
            # Create DataFrame using the exact feature order model was trained on
            input_data = pd.DataFrame([input_dict])[FEATURE_ORDER]

            st.write("**Input Data for Model:**")
            st.dataframe(input_data)

            try:
                # Scale the input data
                input_data_scaled = fitted_scaler.transform(input_data)

                # Make prediction
                prediction = trained_model.predict(input_data_scaled)
                predicted_count = int(round(prediction[0]))

                st.metric("Predicted Daily Bike Rentals", f"{predicted_count:,}")
                if predicted_count < 0:
                    st.warning(
                        "Note: Prediction is negative, suggesting potentially unusual input conditions or model limitations.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.exception(e)
        else:
            st.info("Click the 'Predict Demand' button after adjusting inputs.")

# --- Data Exploration Tab ---
with tab2:
    st.header("Explore Processed Training Data")
    st.write("Data preview (with de-normalized weather features, 'yr' removed):")
    st.dataframe(data_processed_df.head())

    st.divider()  # Visual separator

    c1, c2 = st.columns(2)
    with c1:
        st.write("Average Daily Rentals by Season:")
        seasonal_rentals = data_processed_df.groupby('season')['cnt'].mean().reset_index()
        # Use the display map for labels
        season_map_rev = {v: k for k, v in season_map_display.items()}  # Need codes for mapping
        seasonal_rentals['season_label'] = seasonal_rentals['season'].map(season_map_display)
        st.bar_chart(seasonal_rentals.set_index('season_label')['cnt'])

    with c2:
        st.write("Average Daily Rentals by Weather Situation:")
        weather_rentals = data_processed_df.groupby('weathersit')['cnt'].mean().reset_index()
        weather_rentals['weather_label'] = weather_rentals['weathersit'].map(weather_map)
        st.bar_chart(weather_rentals.set_index('weather_label')['cnt'])