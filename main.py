import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import joblib
import os

# --- Set Page Config FIRST ---
st.set_page_config(layout="wide", page_title="Bike Demand Predictor")

# --- Configuration & Constants ---
DATA_FILE_PATH = "Bike Sharing DC Dataset/day.csv"
# Use new filenames for the model trained with de-normalized data and without 'yr'
MODEL_FILE_PATH = "gradient_boosting_model_v2.joblib"
SCALER_FILE_PATH = "scaler_v2.joblib"
# Add 'yr' to the drop list
FEATURES_TO_DROP = ["cnt", "instant", "dteday", "casual", "registered", "yr"]
# Define original scale calculation constants (from dataset documentation)
T_MIN, T_MAX = -8, 39
AT_MIN, AT_MAX = -16, 50
HUM_MAX = 100
WIND_MAX = 67


# --- Caching Functions ---

def denormalize_data(df):
    """Converts normalized weather features back to original scales."""
    df_copy = df.copy()
    df_copy['temp_c'] = df_copy['temp'] * (T_MAX - T_MIN) + T_MIN
    df_copy['atemp_c'] = df_copy['atemp'] * (AT_MAX - AT_MIN) + AT_MIN
    df_copy['hum_pct'] = df_copy['hum'] * HUM_MAX
    df_copy['windspeed_kmh'] = df_copy['windspeed'] * WIND_MAX
    # Drop original normalized columns after creating new ones
    df_copy = df_copy.drop(columns=['temp', 'atemp', 'hum', 'windspeed'])
    return df_copy


@st.cache_data  # Cache the loaded and processed data
def load_and_prepare_data(file_path):
    """Loads, de-normalizes, and prepares the bike sharing data."""
    if not os.path.exists(file_path):
        st.error(f"Error: Data file not found at {file_path}")
        st.stop()
    try:
        data = pd.read_csv(file_path)
        # De-normalize weather features
        data_processed = denormalize_data(data)

        if 'cnt' not in data_processed.columns:
            st.error("Error: 'cnt' column missing after processing.")
            st.stop()
        return data_processed
    except Exception as e:
        st.error(f"Error loading/processing data: {e}")
        st.stop()


@st.cache_resource  # Cache the trained model and scaler
def load_model_and_scaler(data, features_to_drop):
    """
    Trains (if needed) or loads a Gradient Boosting model and scaler,
    using de-normalized data and excluding 'yr'.
    """
    if os.path.exists(MODEL_FILE_PATH) and os.path.exists(SCALER_FILE_PATH):
        try:
            model = joblib.load(MODEL_FILE_PATH)
            scaler = joblib.load(SCALER_FILE_PATH)
            print("Loaded pre-trained V2 model and scaler.")
            return model, scaler
        except Exception as e:
            st.warning(f"Could not load pre-trained V2 files ({e}). Retraining model...")

    print("Training new V2 model and scaler...")
    try:
        # Use the de-normalized data
        X = data.drop(columns=features_to_drop)
        y = data["cnt"]
        # Check if columns to drop actually exist
        cols_exist = all(item in data.columns for item in features_to_drop if item in X.columns or item == 'cnt')
        if not cols_exist:
            st.error(f"Error: Not all columns to drop {features_to_drop} found in data.")
            st.stop()

        # Ensure feature order is consistent
        feature_names = X.columns.tolist()

        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # --- Scaling ---
        # Scaler is now fitted on de-normalized data (temp_c, hum_pct etc.)
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        # x_val_scaled = scaler.transform(x_val) # Only needed if evaluating here

        # --- Model Training ---
        gb_reg = GradientBoostingRegressor(
            n_estimators=1000, learning_rate=0.1, max_depth=5,
            random_state=42, subsample=0.7
        )
        gb_reg.fit(x_train_scaled, y_train)  # Train on scaled data

        # --- Save the V2 model and scaler ---
        joblib.dump(gb_reg, MODEL_FILE_PATH)
        joblib.dump(scaler, SCALER_FILE_PATH)
        # Also save the feature order used for training
        joblib.dump(feature_names, "feature_names_v2.joblib")

        # --- Optional: Evaluate and Print Performance ---
        x_val_scaled = scaler.transform(x_val)  # Scale validation set for evaluation
        gb_reg_pred = gb_reg.predict(x_val_scaled)
        mse = mean_squared_error(y_val, gb_reg_pred)
        r2 = r2_score(y_val, gb_reg_pred)
        rmse = root_mean_squared_error(y_val, gb_reg_pred)
        print(f"--- Model V2 Performance (Test Set) ---")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"---------------------------------------")

        return gb_reg, scaler

    except Exception as e:
        st.error(f"Error during model training: {e}")
        st.exception(e)  # Print full traceback to Streamlit console
        st.stop()


# --- Load Data and Model ---
data_processed_df = load_and_prepare_data(DATA_FILE_PATH)
# Load feature names used during training (important for prediction consistency)
try:
    FEATURE_ORDER = joblib.load("feature_names_v2.joblib")
except FileNotFoundError:
    # If feature names file doesn't exist, try to infer or force retrain
    st.warning("Feature names file not found. Attempting to infer or retrain.")
    # Force retrain by ensuring model/scaler files don't exist
    if os.path.exists(MODEL_FILE_PATH): os.remove(MODEL_FILE_PATH)
    if os.path.exists(SCALER_FILE_PATH): os.remove(SCALER_FILE_PATH)
    trained_model, fitted_scaler = load_model_and_scaler(data_processed_df, FEATURES_TO_DROP)
    # Try loading feature names again after training should have saved them
    try:
        FEATURE_ORDER = joblib.load("feature_names_v2.joblib")
    except FileNotFoundError:
        st.error("Failed to load feature names even after retraining attempt. Cannot proceed.")
        st.stop()  # Stop execution if feature names cannot be determined

# Now load the model and scaler
trained_model, fitted_scaler = load_model_and_scaler(data_processed_df, FEATURES_TO_DROP)

# --- Streamlit App Layout ---
st.title("🚲 Daily Bike Rental Demand Predictor")
st.markdown("""
Predict the **total number of daily bike rentals** in Washington D.C. based on weather and seasonal conditions.
Adjust the inputs in the sidebar to see the estimated demand.
*(Model: Gradient Boosting Regressor. Note: This model does not consider the specific year.)*
""")

# --- User Input Section ---
st.sidebar.header("Input Conditions:")

# Define input mappings for clarity
season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
# Month names are more user-friendly
month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
             11: 'Nov', 12: 'Dec'}
weekday_map = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
# Updated weather situation descriptions for clarity
weather_map = {1: "Clear / Few Clouds", 2: "Mist / Cloudy", 3: "Light Snow / Light Rain",
               4: "Heavy Rain / Thunderstorm / Snow"}

# Use de-normalized data ranges for sliders
temp_c_min, temp_c_max = float(data_processed_df['temp_c'].min()), float(data_processed_df['temp_c'].max())
atemp_c_min, atemp_c_max = float(data_processed_df['atemp_c'].min()), float(data_processed_df['atemp_c'].max())
hum_pct_min, hum_pct_max = float(data_processed_df['hum_pct'].min()), float(data_processed_df['hum_pct'].max())
wind_kmh_min, wind_kmh_max = float(data_processed_df['windspeed_kmh'].min()), float(
    data_processed_df['windspeed_kmh'].max())

# Create input widgets with better labels and help text
inp_season = st.sidebar.selectbox("Season", options=list(season_map.keys()), format_func=lambda x: season_map[x],
                                  help="Select the season.")
inp_mnth = st.sidebar.selectbox("Month", options=list(month_map.keys()), format_func=lambda x: month_map[x],
                                help="Select the month.")
inp_holiday = st.sidebar.selectbox("Holiday?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                                   help="Is it a public holiday?")
inp_weekday = st.sidebar.selectbox("Day of Week", options=list(weekday_map.keys()),
                                   format_func=lambda x: weekday_map[x], help="Select the day.")
inp_workingday = st.sidebar.selectbox("Working Day?", options=[0, 1],
                                      format_func=lambda x: "No (Weekend/Holiday)" if x == 0 else "Yes (Weekday)",
                                      help="Is it a regular working day (not weekend or holiday)?")
weather_options = sorted(data_processed_df['weathersit'].unique())
inp_weathersit = st.sidebar.selectbox("Weather Situation", options=weather_options,
                                      format_func=lambda x: weather_map.get(x, f"Unknown: {x}"),
                                      help="Select the prevailing weather category for the day.")

# Use number_input for precise temperature, maybe sliders for others
st.sidebar.subheader("Weather Details:")
inp_temp_c = st.sidebar.number_input("Temperature (°C)", min_value=temp_c_min, max_value=temp_c_max,
                                     value=float(data_processed_df['temp_c'].mean()), step=0.5, format="%.1f",
                                     help="Average daily temperature in Celsius.")
inp_atemp_c = st.sidebar.number_input("Feeling Temperature (°C)", min_value=atemp_c_min, max_value=atemp_c_max,
                                      value=float(data_processed_df['atemp_c'].mean()), step=0.5, format="%.1f",
                                      help="'Feels like' temperature in Celsius.")
inp_hum_pct = st.sidebar.slider("Humidity (%)", int(hum_pct_min), int(hum_pct_max),
                                int(data_processed_df['hum_pct'].mean()), help="Average daily humidity in percent.")
inp_windspeed_kmh = st.sidebar.slider("Windspeed (km/h)", int(wind_kmh_min), int(wind_kmh_max),
                                      int(data_processed_df['windspeed_kmh'].mean()),
                                      help="Average daily windspeed in km/h.")

# --- Prediction Logic ---
col1, col2 = st.columns([0.6, 0.4])  # Create columns for layout

with col1:
    st.subheader("Current Input Conditions:")
    # Create input DataFrame using the loaded FEATURE_ORDER
    # This ensures consistency between training and prediction
    input_values = [
        inp_season, inp_mnth, inp_holiday, inp_weekday, inp_workingday, inp_weathersit,
        inp_temp_c, inp_atemp_c, inp_hum_pct, inp_windspeed_kmh
    ]
    input_data = pd.DataFrame([input_values], columns=FEATURE_ORDER)
    st.dataframe(input_data)

with col2:
    st.subheader("Prediction Result:")
    if st.button("Predict Demand", key="predict_button", type="primary", use_container_width=True):
        try:
            # Scale the input data using the loaded V2 scaler
            input_data_scaled = fitted_scaler.transform(input_data)

            # Make prediction
            prediction = trained_model.predict(input_data_scaled)
            predicted_count = int(round(prediction[0]))  # Get the first prediction, round and convert to int

            st.metric("Predicted Daily Bike Rentals", f"{predicted_count:,}")  # Format with comma
            if predicted_count < 0:
                st.warning(
                    "Note: Prediction is negative, suggesting potentially unusual input conditions or model limitations.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.exception(e)
    else:
        st.info("Click the 'Predict Demand' button to see the result.")

# --- Optional: Show some data exploration ---
st.markdown("---")
st.subheader("Explore Processed Data")
expander = st.expander("Click here to explore the data used for training")
with expander:
    st.write("Data preview (with de-normalized weather features):")
    st.dataframe(data_processed_df.head())

    st.write("Average Daily Rentals by Season:")
    seasonal_rentals = data_processed_df.groupby('season')['cnt'].mean().reset_index()
    seasonal_rentals['season_label'] = seasonal_rentals['season'].map(season_map)
    st.bar_chart(seasonal_rentals.set_index('season_label')['cnt'])

    st.write("Average Daily Rentals by Weather Situation:")
    weather_rentals = data_processed_df.groupby('weathersit')['cnt'].mean().reset_index()
    weather_rentals['weather_label'] = weather_rentals['weathersit'].map(weather_map)
    st.bar_chart(weather_rentals.set_index('weather_label')['cnt'])