#Fist Half: Id-23  & Second Half: Id-24

#--------ID:202122104023------Name:Mahadi Hassan--------
#=======================================================
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import random
from datetime import datetime

# ১. ডেটা জেনারেশন ফাংশন
def generate_farmer_data(num=500):
    districts = ['ঢাকা', 'ময়মনসিংহ', 'রংপুর', 'বগুড়া', 'রাজশাহী', 'কুষ্টিয়া', 'বরিশাল', 'সিলেট', 'চট্টগ্রাম', 'খুলনা']
    upazilas = ['সদর', 'গফরগাঁও', 'মিঠাপুকুর', 'শাজাহানপুর', 'চারঘাট', 'মিরপুর', 'আগৈলঝারা', 'বিশ্বম্ভরপুর', 'বোয়ালখালী', 'ডুমুরিয়া']
    irrigation_types = ['নলকূপ', 'নদী', 'বৃষ্টিনির্ভর', 'পাম্প', 'পুকুর']
    farmer_ids = [f'F{1000+i}' for i in range(num)]

    data = []
    for i in range(num):
        district = random.choice(districts)
        upazila = random.choice(upazilas)
        farmer = {
            'farmer_id': farmer_ids[i],
            'name': f'কৃষক_{i+1}',
            'district': district,
            'upazila': upazila,
            'village': f'গ্রাম_{random.randint(1,5)}',
            'land_size': round(random.uniform(0.5, 10.0), 2),  # হেক্টর
            'irrigation_type': random.choice(irrigation_types),
            'tractor_available': random.choice([True, False]),
            'labor_availability': random.choice(['নিম্ন', 'মধ্যম', 'উচ্চ']),
            'investment_capacity': random.choice(['৫০,০০০ টাকা', '১,০০,০০০ টাকা', '২,০০,০০০ টাকা', '৫,০০,০০০ টাকা+']),
            'phone': f'01{random.randint(300000000, 999999999)}'
        }
        data.append(farmer)

    return pd.DataFrame(data)

def generate_soil_data(num=500):
    farmer_ids = [f'F{1000+i}' for i in range(num)]
    data = []
    for fid in farmer_ids:
        soil = {
            'farmer_id': fid,
            'ph_level': round(random.uniform(5.0, 8.5), 1),
            'organic_matter': round(random.uniform(0.5, 3.5), 1),  # %
            'nitrogen': random.randint(50, 200),  # kg/ha
            'phosphorus': random.randint(10, 60),  # ppm
            'potassium': random.randint(80, 300),  # ppm
            'soil_type': random.choice(['বেলে', 'দোআঁশ', 'এটেল', 'পডজল', 'লাল']),
            'soil_moisture': random.randint(20, 80)  # %
        }
        data.append(soil)
    return pd.DataFrame(data)

def generate_market_data():
    crops = ['ধান', 'গম', 'ভুট্টা', 'সয়াবিন', 'আলু', 'পাট', 'মরিচ', 'পেঁয়াজ', 'রসুন', 'সূর্যমুখী',
              'ডাল', 'তিল', 'আখ', 'চা', 'কলা', 'আম', 'পেয়ারা', 'লিচু', 'কাঁঠাল', 'নারিকেল']

    data = []
    for crop in crops:
        base_price = random.randint(500, 5000)
        market = {
            'crop_name': crop,
            'current_price': base_price,  # টাকা/মণ
            'demand_trend': random.choice(['উর্ধ্বমুখী', 'স্থিতিশীল', 'নিম্নমুখী']),
            'export_potential': random.choice([True, False]),
            'local_processing': random.choice(['হ্যাঁ', 'না', 'সীমিত']),
            'price_stability': random.choice(['উচ্চ', 'মধ্যম', 'নিম্ন'])
        }
        data.append(market)
    return pd.DataFrame(data)

def generate_crop_requirements():
    crops = ['ধান', 'গম', 'ভুট্টা', 'সয়াবিন', 'আলু', 'পাট', 'মরিচ', 'পেঁয়াজ', 'রসুন', 'সূর্যমুখী',
             'ডাল', 'তিল', 'আখ', 'চা', 'কলা', 'আম', 'পেয়ারা', 'লিচু', 'কাঁঠাল', 'নারিকেল']

    data = []
    for crop in crops:
        req = {
            'crop_name': crop,
            'min_ph': round(random.uniform(5.0, 7.5), 1),
            'max_ph': round(random.uniform(6.5, 8.5), 1),
            'min_temp': random.randint(15, 25),
            'max_temp': random.randint(25, 35),
            'rainfall_min': random.randint(800, 1500),  # mm
            'rainfall_max': random.randint(1500, 3000),
            'soil_type_preferred': random.choice(['বেলে', 'দোআঁশ', 'এটেল', 'পডজল', 'যেকোনো']),
            'growth_duration': random.randint(90, 180),  # দিন
            'irrigation_needs': random.choice(['নিম্ন', 'মধ্যম', 'উচ্চ'])
        }
        data.append(req)
    return pd.DataFrame(data)

def generate_weather_data(districts):
    seasons = ['রবি', 'খরিপ', 'শীত', 'বর্ষা']
    data = []
    for district in districts:
        for season in seasons:
            weather = {
                'district': district,
                'season': season,
                'avg_temp': random.randint(15, 35),
                'avg_rainfall': random.randint(50, 400),  # mm/month
                'humidity': random.randint(60, 95),  # %
                'sunlight_hours': random.randint(6, 10)  # ঘণ্টা/দিন
            }
            data.append(weather)
    return pd.DataFrame(data)

def generate_seasonal_crops():
    crops = ['ধান', 'গম', 'ভুট্টা', 'সয়াবিন', 'আলু', 'পাট', 'মরিচ', 'পেঁয়াজ', 'রসুন', 'সূর্যমুখী']
    seasons = ['রবি', 'খরিপ', 'শীত', 'বর্ষা']
    districts = ['ঢাকা', 'ময়মনসিংহ', 'রংপুর', 'বগুড়া', 'রাজশাহী', 'কুষ্টিয়া', 'বরিশাল', 'সিলেট', 'চট্টগ্রাম', 'খুলনা']

    data = []
    for district in districts:
        for season in seasons:
            num_crops = random.randint(3, 6)
            selected_crops = random.sample(crops, num_crops)
            for crop in selected_crops:
                seasonal = {
                    'district': district,
                    'season': season,
                    'crop_name': crop,
                    'suitability_score': random.randint(70, 95),
                    'popularity': random.choice(['উচ্চ', 'মধ্যম', 'নিম্ন'])
                }
                data.append(seasonal)
    return pd.DataFrame(data)

# ২. ডেটা জেনারেট এবং সেভ
farmer_df = generate_farmer_data(500)
soil_df = generate_soil_data(500)
market_df = generate_market_data()
crop_req_df = generate_crop_requirements()
weather_df = generate_weather_data(farmer_df['district'].unique())
seasonal_crops_df = generate_seasonal_crops()

# ৩. মডেল ট্রেইনিং এর জন্য ডেটা প্রিপ্রসেসিং
def prepare_training_data():
    # historical yield data তৈরি (সিমুলেটেড)
    historical_data = []
    for _, row in farmer_df.iterrows():
        district = row['district']
        season = random.choice(['রবি', 'খরিপ', 'শীত', 'বর্ষা'])
        available_crops = seasonal_crops_df[
            (seasonal_crops_df['district'] == district) &
            (seasonal_crops_df['season'] == season)
        ]

        if len(available_crops) > 0:
            selected_crop = random.choice(available_crops['crop_name'].tolist())
            soil = soil_df[soil_df['farmer_id'] == row['farmer_id']].iloc[0]
            crop_req = crop_req_df[crop_req_df['crop_name'] == selected_crop].iloc[0]

            # suitability score গণনা
            ph_score = 100 - abs(soil['ph_level'] - (crop_req['min_ph'] + crop_req['max_ph'])/2)*10
            temp_score = 100 - abs(random.randint(15,35) - (crop_req['min_temp'] + crop_req['max_temp'])/2)
            rainfall_score = 100 - abs(random.randint(50,400) - (crop_req['rainfall_min'] + crop_req['rainfall_max'])/2)/5

            suitability = (ph_score*0.4 + temp_score*0.3 + rainfall_score*0.3)

            historical_data.append({
                'farmer_id': row['farmer_id'],
                'district': district,
                'season': season,
                'crop_name': selected_crop,
                'ph_level': soil['ph_level'],
                'organic_matter': soil['organic_matter'],
                'nitrogen': soil['nitrogen'],
                'phosphorus': soil['phosphorus'],
                'potassium': soil['potassium'],
                'soil_type': soil['soil_type'],
                'irrigation_type': row['irrigation_type'],
                'land_size': row['land_size'],
                'suitability_score': suitability,
                'actual_yield': random.randint(50, 100)  # সিমুলেটেড yield (%)
            })

    return pd.DataFrame(historical_data)

historical_df = prepare_training_data()
print(pd.DataFrame(historical_df))
# ৪. মডেল ট্রেইনিং
def train_crop_recommendation_model(df):
    # ফিচার ইঞ্জিনিয়ারিং
    df = df.copy()
    le = LabelEncoder()
    df['soil_type_encoded'] = le.fit_transform(df['soil_type'])
    df['irrigation_type_encoded'] = le.fit_transform(df['irrigation_type'])
    df['district_encoded'] = le.fit_transform(df['district'])
    df['season_encoded'] = le.fit_transform(df['season'])

    # ফিচার সিলেকশন
    features = ['ph_level', 'organic_matter', 'nitrogen', 'phosphorus', 'potassium',
                'soil_type_encoded', 'irrigation_type_encoded', 'district_encoded',
                'season_encoded', 'land_size']
    target = 'crop_name'

    X = df[features]
    y = df[target]

    # ট্রেইন-টেস্ট স্প্লিট
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(X_train.shape)
    # স্ট্যান্ডার্ড স্কেলিং
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # মডেল ট্রেইনিং
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # ইভ্যালুয়েশন
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    return model, scaler, le

model, scaler, label_encoder = train_crop_recommendation_model(historical_df)


#--------ID:202122104024------Name:Mehedi--------
#=======================================================

# ৫. প্রেডিকশন সিস্টেম
class CropRecommendationSystem:
    def __init__(self, model, scaler, label_encoder):
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.current_season = self.get_current_season()

    def get_current_season(self):
        today = datetime.now()
        month = today.month

        if month in [11, 12, 1]:
            return 'শীত'
        elif month in [2, 3, 4]:
            return 'বর্ষা'
        elif month in [5, 6, 7]:
            return 'খরিপ'
        else:
            return 'রবি'

    def preprocess_farmer_data(self, farmer_data, soil_data):
        # ডেটা মিলানো
        farmer_soil = pd.merge(farmer_data, soil_data, on='farmer_id')

        # এনকোডিং
        farmer_soil['soil_type_encoded'] = self.label_encoder.fit_transform(farmer_soil['soil_type'])
        farmer_soil['irrigation_type_encoded'] = self.label_encoder.fit_transform(farmer_soil['irrigation_type'])
        farmer_soil['district_encoded'] = self.label_encoder.fit_transform(farmer_soil['district'])

        # বর্তমান ঋতু যোগ করা
        farmer_soil['season'] = self.current_season
        farmer_soil['season_encoded'] = self.label_encoder.fit_transform(farmer_soil['season'])

        return farmer_soil

    def predict_crop(self, farmer_data, soil_data, market_data, seasonal_crops):
        # ডেটা প্রিপ্রসেস
        farmer_soil = self.preprocess_farmer_data(farmer_data, soil_data)

        # ফিচার সিলেকশন
        features = ['ph_level', 'organic_matter', 'nitrogen', 'phosphorus', 'potassium',
                    'soil_type_encoded', 'irrigation_type_encoded', 'district_encoded',
                    'season_encoded', 'land_size']

        X = farmer_soil[features]
        X_scaled = self.scaler.transform(X)

        # প্রেডিকশন
        probabilities = self.model.predict_proba(X_scaled)
        crops = self.model.classes_

        # ফলাফল প্রস্তুত
        recommendations = []
        for i in range(len(farmer_soil)):
            farmer_id = farmer_soil.iloc[i]['farmer_id']
            district = farmer_soil.iloc[i]['district']

            # শুধুমাত্র বর্তমান ঋতু এবং জেলার জন্য উপযুক্ত ফসল বিবেচনা
            available_crops = seasonal_crops[
                (seasonal_crops['district'] == district) &
                (seasonal_crops['season'] == self.current_season)
            ]['crop_name'].tolist()

            crop_probs = []
            for j, crop in enumerate(crops):
                if crop in available_crops:
                    market_info = market_data[market_data['crop_name'] == crop].iloc[0]
                    crop_probs.append({
                        'crop': crop,
                        'probability': probabilities[i][j],
                        'market_price': market_info['current_price'],
                        'demand_trend': market_info['demand_trend']
                    })

            # সম্ভাব্যতা এবং বাজার মূল্য ভিত্তিতে সাজান
            crop_probs.sort(key=lambda x: (x['probability']*0.7 + x['market_price']*0.3), reverse=True)

            # শীর্ষ ৩ সুপারিশ নিন
            top_recommendations = crop_probs[:3]

            recommendations.append({
                'farmer_id': farmer_id,
                'district': district,
                'season': self.current_season,
                'recommendations': top_recommendations
            })

        return recommendations

    def get_crop_details(self, crop_name):
        crop_info = crop_req_df[crop_req_df['crop_name'] == crop_name].iloc[0]
        market_info = market_df[market_df['crop_name'] == crop_name].iloc[0]

        return {
            'crop_name': crop_name,
            'ph_range': f"{crop_info['min_ph']}-{crop_info['max_ph']}",
            'temperature_range': f"{crop_info['min_temp']}°C-{crop_info['max_temp']}°C",
            'rainfall_requirements': f"{crop_info['rainfall_min']}-{crop_info['rainfall_max']} mm",
            'soil_type_preferred': crop_info['soil_type_preferred'],
            'growth_duration': f"{crop_info['growth_duration']} দিন",
            'irrigation_needs': crop_info['irrigation_needs'],
            'market_price': f"{market_info['current_price']} টাকা/মণ",
            'demand_trend': market_info['demand_trend'],
            'export_potential': market_info['export_potential']
        }

# ৬. সিস্টেম ইনিশিয়ালাইজ এবং ব্যবহার
recommender = CropRecommendationSystem(model, scaler, label_encoder)

# একটি নির্দিষ্ট কৃষকের জন্য সুপারিশ
sample_farmer = farmer_df[farmer_df['farmer_id'] == 'F1000']
sample_soil = soil_df[soil_df['farmer_id'] == 'F1000']

recommendations = recommender.predict_crop(sample_farmer, sample_soil, market_df, seasonal_crops_df)

# ফলাফল প্রদর্শন
print("\nকৃষক ID:", recommendations[0]['farmer_id'])
print("জেলা:", recommendations[0]['district'])
print("মৌসুম:", recommendations[0]['season'])
print("\nশীর্ষ ফসল সুপারিশ:")

for idx, rec in enumerate(recommendations[0]['recommendations'], 1):
    crop_details = recommender.get_crop_details(rec['crop'])
    print(f"\n{idx}. {rec['crop']}")
    print(f"   সম্ভাব্যতা: {rec['probability']:.2%}")
    print(f"   বাজার মূল্য: {rec['market_price']} টাকা/মণ")
    print(f"   চাহিদা প্রবণতা: {rec['demand_trend']}")
    print(f"   pH রেঞ্জ: {crop_details['ph_range']}")
    print(f"   মাটির ধরন: {crop_details['soil_type_preferred']}")
    print(f"   সেচ প্রয়োজন: {crop_details['irrigation_needs']}")

# ৭. সমস্ত ডেটা CSV তে সেভ করা
farmer_df.to_csv('farmers_data.csv', index=False, encoding='utf-8-sig')
soil_df.to_csv('soil_data.csv', index=False, encoding='utf-8-sig')
market_df.to_csv('market_data.csv', index=False, encoding='utf-8-sig')
crop_req_df.to_csv('crop_requirements.csv', index=False, encoding='utf-8-sig')
weather_df.to_csv('weather_data.csv', index=False, encoding='utf-8-sig')
seasonal_crops_df.to_csv('seasonal_crops.csv', index=False, encoding='utf-8-sig')
historical_df.to_csv('historical_yield_data.csv', index=False, encoding='utf-8-sig')

# ৮. মডেল সেভ করা
joblib.dump(model, 'crop_recommendation_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

print("\nসমস্ত ডেটা এবং মডেল সফলভাবে সেভ করা হয়েছে!")