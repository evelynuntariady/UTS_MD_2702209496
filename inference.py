import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('RF_hotel.pkl')
encoder = joblib.load('oneHot_encode.pkl')
scaler_std = joblib.load('normdist_normalization.pkl')
scaler_rob = joblib.load('nonnormdist_normalization.pkl')

onnecol = ["type_of_meal_plan", "room_type_reserved", "market_segment_type"]
normdist = ['no_of_adults', 'arrival_month', 'arrival_date']
nonnormdist = ['no_of_children', 'no_of_week_nights', 'lead_time', 'arrival_year', 'no_of_previous_cancellations', 'avg_price_per_room', 'no_of_previous_bookings_not_canceled', 'no_of_special_requests']

def preprocess_input(features):
    df = pd.DataFrame([features], columns=[
        "no_of_adults", "no_of_children", "no_of_week_nights", "no_of_weekend_nights",
        "type_of_meal_plan", "room_type_reserved", "required_car_parking_space",
        "lead_time", "arrival_year", "arrival_month", "arrival_date",
        "market_segment_type", "repeated_guest", "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled", "avg_price_per_room",
        "no_of_special_requests"
    ])
    
    df['required_car_parking_space'] = df['required_car_parking_space'].map({'Ya': 1, 'Tidak': 0})
    df['repeated_guest'] = df['repeated_guest'].map({'Ya': 1, 'Tidak': 0})
    
    encoded_cols = encoder.transform(df[onnecol])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(onnecol))

    
    final_df = pd.concat([df, encoded_df], axis=1)
    final_df = final_df.drop(columns=onnecol)

    final_df[normdist] = scaler_std.transform(final_df[normdist])
    final_df[nonnormdist] = scaler_rob.transform(final_df[nonnormdist])
    
    return final_df

def main():
    st.title('Booking Status Classification Model Deployment')
    no_of_adults=st.number_input("Jumlah orang dewasa", 0, 100)
    no_of_children=st.number_input("Jumlah anak kecil", 0, 100)
    no_of_weekend_nights=st.number_input("Jumlah malam akhir pekan", 0, 100)
    no_of_week_nights=st.number_input("Jumlah malam dalam seminggu", 0, 100)
    type_of_meal_plan=st.radio("Jenis paket makanan", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
    room_type_reserved=st.radio("Jenis kamar yang dipesan", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6' ,'Room_Type 7'])
    required_car_parking_space=st.radio("Apakah pelanggan membutuhkan tempat parkir mobil?",['Ya', 'Tidak'])
    lead_time=st.slider("Jumlah hari antara tanggal pemesanan dan tanggal kedatangan", 0,365)
    arrival_year=st.radio("Tahun tanggal kedatangan", [2017,2018])
    arrival_month=st.slider("Bulan tanggal kedatangan", 1,12)
    arrival_date=st.number_input("Tanggal kedatangan", 0, 31)
    market_segment_type = st.radio('Jenis segmen pasar', ['Online' ,'Offline', 'Corporate', 'Complementary', 'Aviation'])
    repeated_guest = st.radio("Apakah pelanggan tersebut merupakan tamu yang pernah melakukan booking dan juga menginap?",['Ya', 'Tidak'])
    no_previous_cancellations = st.number_input('Jumlah pemesanan yang dibatalkan sebelumnya', 0, 20)
    no_of_previous_bookings_not_canceled = st.number_input('Jumlah pemesanan yang tidak dibatalkan sebelumnya', 0, 60)
    avg_price_per_room = st.number_input('Harga rata-rata per hari pemesanan', 0, 540)
    no_of_special_requests = st.number_input('Jumlah total permintaan khusus', 0, 5)
    
    
    if st.button('Make Prediction'):
        features = [no_of_adults, no_of_children, no_of_week_nights, no_of_weekend_nights, type_of_meal_plan,
                    room_type_reserved, required_car_parking_space, lead_time, arrival_year,
                    arrival_month, arrival_date, market_segment_type, repeated_guest, no_previous_cancellations,
                    no_of_previous_bookings_not_canceled, avg_price_per_room, no_of_special_requests]
        input_final = preprocess_input(features)
        result = make_prediction(input_final)
        label = 'Canceled' if result == 1 else 'Not Canceled'
        st.success(f'The booking status is: {label}')

def make_prediction(features):

    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()

