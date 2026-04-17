import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Nepal House Construction Cost Predictor",
    page_icon="🏠",
    layout="centered"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load models and encoders
@st.cache_resource
def load_models():
    try:
        model = joblib.load('./thesis_figures/random_forest_model.pkl')
        loc_enc = joblib.load('./thesis_figures/location_encoder.pkl')
        found_enc = joblib.load('./thesis_figures/foundation_encoder.pkl')
        return model, loc_enc, found_enc
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

rf_model, le_location, le_foundation = load_models()

def main():
    st.title("🏠 Nepal Building Cost Predictor")
    st.write("Enter the specifications of the building below to estimate the total construction cost.")
    st.divider()

    if rf_model is None:
        st.warning("Please ensure 'random_forest_model.pkl', 'location_encoder.pkl', and 'foundation_encoder.pkl' are in the application directory.")
        return

    # Layout using columns for a compact UI
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Physical Dimensions")
        plinth_area = st.number_input("Plinth Area (sq. ft.)", min_value=100.0, max_value=10000.0, value=1200.0)
        no_of_storeys = st.number_input("Number of Storeys", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
        no_of_columns = st.number_input("Number of Columns", min_value=4, max_value=100, value=12)

    with col2:
        st.subheader("Categorical Details")
        location = st.selectbox("Location", options=le_location.classes_)
        foundation = st.selectbox("Foundation Type", options=le_foundation.classes_)

    st.divider()

    # Calculation and Prediction
    if st.button("Calculate Predicted Cost"):
        # 1. Feature Engineering (Match the training logic)
        total_area = plinth_area * no_of_storeys
        
        # 2. Encoding
        try:
            loc_idx = le_location.transform([location])[0]
            found_idx = le_foundation.transform([foundation])[0]
            
            # 3. Form Input Vector
            # Order based on your training script: 
            # ["Total_Area", "No. of Storeys", "Plinth Area", "No. of Columns", "Location_Enc", "Found_Enc"]
            input_data = np.array([[
                total_area, 
                no_of_storeys, 
                plinth_area, 
                no_of_columns, 
                loc_idx, 
                found_idx
            ]])

            # 4. Prediction
            prediction = rf_model.predict(input_data)[0]

            # Display Results
            st.markdown(f"""
                <div class="result-card">
                    <h2 style='color: #28a745;'>Estimated Total Cost</h2>
                    <h1 style='color: #1e7e34;'>NPR {prediction:,.2f}</h1>
                    <p style='color: #6c757d;'>Total Built-up Area: {total_area:,.2f} sq. ft.</p>
                    <p style='color: #6c757d;'>Estimated Cost per sq. ft.: NPR {prediction/total_area:,.2f}</p>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

    # Footer
    st.markdown("---")

if __name__ == "__main__":
    main()