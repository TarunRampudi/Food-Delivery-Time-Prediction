
import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🍔",
    layout="wide"
)

st.title("🍔 Food Delivery Time Prediction")
st.write("Predict the estimated food delivery time using Tuned XGBoost.")

API_URL = "http://127.0.0.1:8000/predict"

st.subheader("Enter Delivery Details")

with st.form("delivery_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Delivery Person Details")

        age = st.number_input(
            "Delivery Person Age", 18, 60, 37
        )

        ratings = st.number_input(
            "Delivery Person Ratings",
            1.0, 5.0, 4.9, step=0.1
        )

        vehicle_condition = st.selectbox(
            "Vehicle Condition", [0, 1, 2, 3], index=2
        )

        multiple_deliveries = st.selectbox(
            "Multiple Deliveries", [0, 1, 2, 3], index=0
        )

        vehicle = st.selectbox(
            "Type of Vehicle", ["motorcycle", "scooter"]
        )

    with col2:
        st.markdown("### Order Details")

        order_date = st.text_input(
            "Order Date (DD-MM-YYYY)", "19-03-2022"
        )

        order_time = st.text_input(
            "Order Time (HH:MM:SS)", "11:33:33"
        )

        pickup_time = st.text_input(
            "Pickup Time (HH:MM:SS)", "11:45:29"
        )

        order_type = st.selectbox(
            "Type of Order", ["Snack", "Meal", "Drinks", "Buffet"]
        )

        weather = st.selectbox(
            "Weather Conditions",
            [
                "conditions Sunny",
                "conditions Stormy",
                "conditions Sandstorms",
                "conditions Cloudy",
                "conditions Windy",
                "conditions Fog"
            ]
        )

        traffic = st.selectbox(
            "Road Traffic Density", ["Low", "Medium", "High", "Jam"]
        )

        festival = st.selectbox("Festival", ["No", "Yes"])

        city = st.selectbox(
            "City", ["Urban", "Metropolitian", "Semi-Urban"]
        )

    st.markdown("### Restaurant and Delivery Location")

    col3, col4 = st.columns(2)

    with col3:
        restaurant_latitude = st.number_input(
            "Restaurant Latitude", value=22.745049, format="%.6f"
        )
        restaurant_longitude = st.number_input(
            "Restaurant Longitude", value=75.892471, format="%.6f"
        )

    with col4:
        delivery_latitude = st.number_input(
            "Delivery Latitude", value=22.765049, format="%.6f"
        )
        delivery_longitude = st.number_input(
            "Delivery Longitude", value=75.912471, format="%.6f"
        )

    submitted = st.form_submit_button(
        "Predict Delivery Time",
        use_container_width=True
    )

if submitted:

    input_data = {
        "Delivery_person_Age": age,
        "Delivery_person_Ratings": ratings,
        "Restaurant_latitude": restaurant_latitude,
        "Restaurant_longitude": restaurant_longitude,
        "Delivery_location_latitude": delivery_latitude,
        "Delivery_location_longitude": delivery_longitude,
        "Order_Date": order_date,
        "Time_Orderd": order_time,
        "Time_Order_picked": pickup_time,
        "Weatherconditions": weather,
        "Road_traffic_density": traffic,
        "Vehicle_condition": vehicle_condition,
        "Type_of_order": order_type,
        "Type_of_vehicle": vehicle,
        "multiple_deliveries": multiple_deliveries,
        "Festival": festival,
        "City": city
    }

    try:
        response = requests.post(
            API_URL,
            json=input_data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()

            st.success("Prediction completed successfully!")

            st.metric(
                "Predicted Delivery Time",
                f"{result['predicted_delivery_time_minutes']} minutes"
            )

        else:
            st.error(f"API Error: {response.status_code}")
            st.write(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. "
            "Please start the FastAPI server first."
        )

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
