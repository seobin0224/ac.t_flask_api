import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pymongo
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from datetime import datetime, timedelta
import json
import schedule
import time
from dotenv import load_dotenv
import os
import re
from collections import Counter

# 환경 변수에서 MONGO_URI 가져오기
mongo_uri = os.getenv('MONGO_URI')

class ActivityRecommendationModel(nn.Module):
    def __init__(self, weather_input_size, climate_input_size, hidden_size, num_activities):
        super(ActivityRecommendationModel, self).__init__()
        self.weather_fc = nn.Linear(weather_input_size, hidden_size)
        self.climate_fc = nn.Linear(climate_input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_activities)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, weather, climate):
        weather_features = self.relu(self.weather_fc(weather))
        climate_features = self.relu(self.climate_fc(climate))
        combined = torch.cat((weather_features, climate_features), dim=1)
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    climate_data_path = os.path.join(base_path, 'gangwon_climate_data.json')
    activity_data_path = os.path.join(base_path, 'train_data_weather.json')
    
    with open(climate_data_path, 'r') as f:
        climate_data = json.load(f)
    
    with open(activity_data_path, 'r') as f:
        activity_data = json.load(f)
    
    # 위치 이름 전처리
    climate_data = {preprocess_location_name(k): v for k, v in climate_data.items()}
    
    # 활동 이름 전처리
    for activity in activity_data:
        activity['name'] = preprocess_activity_name(activity['name'])
    
    return climate_data, activity_data

def preprocess_location_name(name):
    # Remove '시', '군', '구' at the end of the name
    name = re.sub(r'(시|군|구)$', '', name)
    return name

def preprocess_activity_name(name):
    # 활동 이름에서 모든 공백 제거
    return ''.join(name.split())

def preprocess_data(climate_data, activity_data):
    X_activity = []
    y = []
    X_climate = []
    
    sky_status_map = {'맑음': [1, 0, 0], '구름 많음': [0, 1, 0], '흐림': [0, 0, 1]}
    precip_type_map = {'없음': [1, 0, 0, 0, 0], '비': [0, 1, 0, 0, 0], '비/눈': [0, 0, 1, 0, 0], '눈': [0, 0, 0, 1, 0], '소나기': [0, 0, 0, 0, 1]}
    
    for activity in activity_data:
        for city, climate in climate_data.items():
            sky_status_encoded = sky_status_map[activity['sky_status']]
            precip_type_encoded = precip_type_map[activity['precipitation']]
            
            X_activity.append([
                float(activity['temperature_min']),
                float(activity['temperature_max']),
                float(activity['humidity']),
                float(activity['wind_speed']),
                *sky_status_encoded,
                *precip_type_encoded
            ])
            X_climate.append([
                climate['winter_temp'],
                climate['summer_temp'],
                climate['annual_precipitation'],
                climate['winter_precipitation'],
                climate['summer_precipitation'],
                climate['snow_days'],
                climate['foggy_days']
            ])
            y.append(activity_data.index(activity))
    
    weather_scaler = StandardScaler()
    climate_scaler = StandardScaler()
    X_activity_normalized = weather_scaler.fit_transform(X_activity)
    X_climate_normalized = climate_scaler.fit_transform(X_climate)

    return torch.tensor(X_activity_normalized, dtype=torch.float32), torch.tensor(X_climate_normalized, dtype=torch.float32), torch.tensor(y, dtype=torch.long), weather_scaler, climate_scaler

def train_model(X_weather, X_climate, y, num_epochs=100, learning_rate=0.001):
    X_train, X_val, y_train, y_val = train_test_split(X_weather, y, test_size=0.2, random_state=42)
    climate_train, climate_val, _, _ = train_test_split(X_climate, y, test_size=0.2, random_state=42)

    weather_input_size = X_weather.shape[1]
    climate_input_size = X_climate.shape[1]
    hidden_size = 64
    num_activities = len(torch.unique(y))

    model = ActivityRecommendationModel(weather_input_size, climate_input_size, hidden_size, num_activities)
    
    # 클래스 가중치 계산
    y_np = y.numpy()
    class_counts = Counter(y_np)
    total_samples = len(y_np)
    class_weights = torch.zeros(num_activities)
    for cls in range(num_activities):
        if cls in class_counts:
            class_weights[cls] = total_samples / class_counts[cls]
        else:
            class_weights[cls] = 1.0  # 데이터가 없는 클래스에 대한 기본 가중치

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train, climate_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val, climate_val)
                val_loss = criterion(val_outputs, y_val)
                _, predicted = torch.max(val_outputs, 1)
                accuracy = (predicted == y_val).float().mean()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {accuracy.item():.4f}')

    return model

def save_model(model, weather_scaler, climate_scaler, filename='src/activity_recommendation_model.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'weather_scaler': weather_scaler,
        'climate_scaler': climate_scaler
    }, filename)

def load_trained_model(filename='activity_recommendation_model.pth'):
    try:
        # 절대 경로 사용
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found at {file_path}")
        
        checkpoint = torch.load(file_path)
        
        weather_input_size = checkpoint['weather_scaler'].n_features_in_
        climate_input_size = checkpoint['climate_scaler'].n_features_in_
        hidden_size = 64  # 이 값이 모델 구조에 맞는지 확인 필요
        num_activities = len(json.load(open(os.path.join(base_path, 'train_data_weather.json'), 'r')))

        model = ActivityRecommendationModel(weather_input_size, climate_input_size, hidden_size, num_activities)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"Model loaded successfully from {file_path}")
        return model, checkpoint['weather_scaler'], checkpoint['climate_scaler']
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def get_weather_forecast_from_mongodb():
    db = get_db()
    short_forecast = db['shortweatherdatas']
    mid_forecast = db['forecasts']
    
    weather_data = {}
    current_date = datetime.now().date()

    # 단기 예보 처리 (3일)
    for doc in short_forecast.find():
        city = preprocess_location_name(doc['locationTag'])
        if city not in weather_data:
            weather_data[city] = {}
        
        for weather in doc['weatherInfo']:
            date = datetime.strptime(weather['date'], '%Y-%m-%d').date()
            if (date - current_date).days <= 2:
                weather_data[city][str(date)] = {
                    'temperature_min': float(weather['temperature']) - 5,
                    'temperature_max': float(weather['temperature']) + 5,
                    'humidity': float(weather['humidity']),
                    'wind_speed': float(weather['windSpeed']),
                    'precipitation': weather['precipitationType'],
                    'sky_status': weather['skyStatus']
                }

    # 중기 예보 처리 (7일)
    for doc in mid_forecast.find():
        city = preprocess_location_name(doc['city'])
        if city not in weather_data:
            weather_data[city] = {}
        
        start_date = current_date + timedelta(days=3)
        for i, (temp, land) in enumerate(zip(doc['temperature'], doc['landForecast'])):
            date = start_date + timedelta(days=i)
            if (date - current_date).days <= 9:
                weather_data[city][str(date)] = {
                    'temperature_min': temp['min'],
                    'temperature_max': temp['max'],
                    'humidity': 70,
                    'wind_speed': 5,
                    'precipitation': '비' if '비' in land['morning'] or '비' in land['evening'] else '없음',
                    'sky_status': '맑음' if '맑음' in land['morning'] or '맑음' in land['evening'] else '흐림'
                }

    return weather_data

def get_db():
    try:
        client = MongoClient(mongo_uri)
        client.server_info()  # 연결 테스트
        return client['actapp']
    except ServerSelectionTimeoutError as err:
        print(f"MongoDB 연결 오류: {err}")
        print(f"현재 MONGO_URI: {mongo_uri}")
        raise

def get_user_activity_data():
    db = get_db()
    user_activities = db['useractivities']
    
    # 모든 사용자 활동 데이터 가져오기
    all_activities = list(user_activities.find())
    
    # 위치 이름 전처리 적용
    for activity in all_activities:
        activity['location'] = preprocess_location_name(activity['location'])
    
    return all_activities

def recommend_activity(model, weather, climate, weather_scaler, climate_scaler, activity_data):
    # Encode sky status
    sky_status_encoded = [0, 0, 0]  # [맑음, 구름 많음, 흐림]
    if weather['sky_status'] == '맑음':
        sky_status_encoded[0] = 1
    elif weather['sky_status'] == '구름 많음':
        sky_status_encoded[1] = 1
    else:
        sky_status_encoded[2] = 1

    # Encode precipitation type
    precip_type_encoded = [0, 0, 0, 0, 0]  # [없음, 비, 비/눈, 눈, 소나기]
    if weather['precipitation'] == '없음':
        precip_type_encoded[0] = 1
    elif weather['precipitation'] == '비':
        precip_type_encoded[1] = 1
    elif weather['precipitation'] == '비/눈':
        precip_type_encoded[2] = 1
    elif weather['precipitation'] == '눈':
        precip_type_encoded[3] = 1
    elif weather['precipitation'] == '소나기':
        precip_type_encoded[4] = 1

    weather_input = np.array([[
        float(weather['temperature_min']),
        float(weather['temperature_max']),
        float(weather['humidity']),
        float(weather['wind_speed']),
        *sky_status_encoded,
        *precip_type_encoded
    ]])
    
    climate_input = np.array([[
        float(climate['winter_temp']),
        float(climate['summer_temp']),
        float(climate['annual_precipitation']),
        float(climate['winter_precipitation']),
        float(climate['summer_precipitation']),
        float(climate['snow_days']),
        float(climate['foggy_days'])
    ]])
    
    weather_normalized = weather_scaler.transform(weather_input)
    climate_normalized = climate_scaler.transform(climate_input)
    
    weather_tensor = torch.tensor(weather_normalized, dtype=torch.float32)
    climate_tensor = torch.tensor(climate_normalized, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(weather_tensor, climate_tensor)
        scores = output[0].numpy()
    
    # 모든 활동에 대한 추천 점수 계산
    recommended_activities = [
        {'activity': activity['name'], 'score': float(scores[i])}
        for i, activity in enumerate(activity_data)
    ]
    
    # 점수에 따라 내림차순 정렬
    recommended_activities.sort(key=lambda x: x['score'], reverse=True)
    
    return recommended_activities

def recommend_activities_for_location(model, location, climate_data, weather_forecast, weather_scaler, climate_scaler, activity_data):
    location = preprocess_location_name(location)
    if location not in climate_data:
        raise ValueError(f"Location '{location}' not found in climate data")

    climate = climate_data[location]
    weather = weather_forecast[location][list(weather_forecast[location].keys())[0]]  # 첫 번째 날짜의 날씨 사용

    weather_input = prepare_weather_input(weather)
    climate_input = prepare_climate_input(climate)

    weather_normalized = weather_scaler.transform(weather_input)
    climate_normalized = climate_scaler.transform(climate_input)

    weather_tensor = torch.tensor(weather_normalized, dtype=torch.float32)
    climate_tensor = torch.tensor(climate_normalized, dtype=torch.float32)

    with torch.no_grad():
        output = model(weather_tensor, climate_tensor)
        scores = output[0].numpy()

    activities_with_scores = [(activity['name'], float(score)) for activity, score in zip(activity_data, scores)]
    activities_with_scores.sort(key=lambda x: x[1], reverse=True)

    return activities_with_scores

def recommend_locations_for_activity(model, activity, climate_data, weather_forecast, weather_scaler, climate_scaler, activity_data):
    activity_index = next((i for i, a in enumerate(activity_data) if a['name'] == activity), None)
    if activity_index is None:
        raise ValueError(f"Activity '{activity}' not found in activity data")

    locations_with_scores = []

    for location, climate in climate_data.items():
        weather = weather_forecast[location][list(weather_forecast[location].keys())[0]]  # 첫 번째 날짜의 날씨 사용

        weather_input = prepare_weather_input(weather)
        climate_input = prepare_climate_input(climate)

        weather_normalized = weather_scaler.transform(weather_input)
        climate_normalized = climate_scaler.transform(climate_input)

        weather_tensor = torch.tensor(weather_normalized, dtype=torch.float32)
        climate_tensor = torch.tensor(climate_normalized, dtype=torch.float32)

        with torch.no_grad():
            output = model(weather_tensor, climate_tensor)
            score = output[0][activity_index].item()

        locations_with_scores.append((location, float(score)))

    locations_with_scores.sort(key=lambda x: x[1], reverse=True)

    return locations_with_scores

def recommend_dates_for_activity_and_location(model, activity, location, climate_data, weather_forecast, weather_scaler, climate_scaler, activity_data):
    activity_index = next((i for i, a in enumerate(activity_data) if a['name'] == activity), None)
    if activity_index is None:
        raise ValueError(f"Activity '{activity}' not found in activity data")

    location = preprocess_location_name(location)
    if location not in climate_data:
        raise ValueError(f"Location '{location}' not found in climate data")

    climate = climate_data[location]
    dates_with_scores = []

    for date, weather in weather_forecast[location].items():
        weather_input = prepare_weather_input(weather)
        climate_input = prepare_climate_input(climate)

        weather_normalized = weather_scaler.transform(weather_input)
        climate_normalized = climate_scaler.transform(climate_input)

        weather_tensor = torch.tensor(weather_normalized, dtype=torch.float32)
        climate_tensor = torch.tensor(climate_normalized, dtype=torch.float32)

        with torch.no_grad():
            output = model(weather_tensor, climate_tensor)
            score = output[0][activity_index].item()

        dates_with_scores.append((date, float(score)))

    dates_with_scores.sort(key=lambda x: x[1], reverse=True)

    return dates_with_scores[:10]  # 상위 10개 날짜 반환

def prepare_weather_input(weather):
    sky_status_encoded = [0, 0, 0]
    if weather['sky_status'] == '맑음':
        sky_status_encoded[0] = 1
    elif weather['sky_status'] == '구름 많음':
        sky_status_encoded[1] = 1
    else:
        sky_status_encoded[2] = 1

    precip_type_encoded = [0, 0, 0, 0, 0]
    if weather['precipitation'] == '없음':
        precip_type_encoded[0] = 1
    elif weather['precipitation'] == '비':
        precip_type_encoded[1] = 1
    elif weather['precipitation'] == '비/눈':
        precip_type_encoded[2] = 1
    elif weather['precipitation'] == '눈':
        precip_type_encoded[3] = 1
    elif weather['precipitation'] == '소나기':
        precip_type_encoded[4] = 1

    return np.array([[
        float(weather['temperature_min']),
        float(weather['temperature_max']),
        float(weather['humidity']),
        float(weather['wind_speed']),
        *sky_status_encoded,
        *precip_type_encoded
    ]])

def prepare_climate_input(climate):
    return np.array([[
        float(climate['winter_temp']),
        float(climate['summer_temp']),
        float(climate['annual_precipitation']),
        float(climate['winter_precipitation']),
        float(climate['summer_precipitation']),
        float(climate['snow_days']),
        float(climate['foggy_days'])
    ]])

def preprocess_user_data_initial(user_activities, climate_data, activity_data):
    X_weather = []
    X_climate = []
    y_new = []
    
    location_mapping = {loc: i for i, loc in enumerate(climate_data.keys())}
    activity_mapping = {preprocess_activity_name(activity['name']): i for i, activity in enumerate(activity_data)}
    
    for activity in user_activities:
        location = preprocess_location_name(activity['location'])
        if location in climate_data:
            climate = climate_data[location]
            weather = activity['weather']
            
            try:
                sky_status_encoded = [0, 0, 0]
                sky_status_encoded[['맑음', '구름 많음', '흐림'].index(weather['sky_status'])] = 1
                
                precip_type_encoded = [0, 0, 0, 0, 0]
                precip_type_encoded[['없음', '비', '비/눈', '눈', '소나기'].index(weather['precipitation'])] = 1
                
                X_weather.append([
                    float(weather['temperature_min']),
                    float(weather['temperature_max']),
                    float(weather['humidity']),
                    float(weather['wind_speed']),
                    *sky_status_encoded,
                    *precip_type_encoded
                ])
                
                X_climate.append([
                    float(climate['winter_temp']),
                    float(climate['summer_temp']),
                    float(climate['annual_precipitation']),
                    float(climate['winter_precipitation']),
                    float(climate['summer_precipitation']),
                    float(climate['snow_days']),
                    float(climate['foggy_days'])
                ])
                
                processed_activity = preprocess_activity_name(activity['activityTag'])
                y_new.append(activity_mapping[processed_activity])
            except (ValueError, KeyError) as e:
                print(f"데이터 처리 중 오류 발생: {str(e)}. 해당 데이터를 건너뜁니다.")
                continue
    
    return np.array(X_weather), np.array(X_climate), np.array(y_new)

def initial_batch_learning():
    print("초기 배치 학습을 시작합니다...")
    global model, weather_scaler, climate_scaler

    try:
        # 기존 데이터 로드
        climate_data, activity_data = load_data()
        
        try:
            # 현재 모델과 스케일러 로드 시도
            model, weather_scaler, climate_scaler = load_trained_model()
        except FileNotFoundError:
            print("기존 모델을 찾을 수 없습니다. 새 모델을 초기화합니다.")
            # 새로운 모델 및 스케일러 초기화
            X_weather, X_climate, y, weather_scaler, climate_scaler = preprocess_data(climate_data, activity_data)
            model = train_model(X_weather, X_climate, y)
        
        # useractivities에서 모든 사용자 활동 데이터 가져오기
        user_activities = get_user_activity_data()
        
        # 사용자 데이터 전처리
        X_weather, X_climate, y_new = preprocess_user_data_initial(user_activities, climate_data, activity_data)
        
        if len(X_weather) > 0:
            # 데이터 정규화 (기존 스케일러 사용)
            X_weather_normalized = weather_scaler.transform(X_weather)
            X_climate_normalized = climate_scaler.transform(X_climate)
            
            # 모델 재학습
            X_weather_tensor = torch.tensor(X_weather_normalized, dtype=torch.float32)
            X_climate_tensor = torch.tensor(X_climate_normalized, dtype=torch.float32)
            y_tensor = torch.tensor(y_new, dtype=torch.long)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            model.train()
            for epoch in range(50):  # 에폭 수는 조정 가능
                optimizer.zero_grad()
                outputs = model(X_weather_tensor, X_climate_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f"에폭 [{epoch+1}/50], 손실: {loss.item():.4f}")
            
            # 모델 저장
            save_model(model, weather_scaler, climate_scaler)
            
            print(f"초기 배치 학습이 완료되었습니다. {len(X_weather)}개의 데이터 포인트로 모델이 업데이트되었습니다.")
        else:
            print("초기 배치 학습을 위한 데이터가 없습니다.")
        
        return model, weather_scaler, climate_scaler, True
    except Exception as e:
        print(f"초기 배치 학습 중 오류가 발생했습니다: {str(e)}")
        # 오류 발생 시에도 모델, 스케일러를 반환
        return model, weather_scaler, climate_scaler, False

def preprocess_user_data(user_activities, climate_data, activity_data):
    X_weather = []
    X_climate = []
    y_new = []
    
    activity_mapping = {preprocess_activity_name(activity['name']): i for i, activity in enumerate(activity_data)}
    
    for activity in user_activities:
        location = preprocess_location_name(activity['location'])
        if location in climate_data:
            climate = climate_data[location]
            
            # 활동 날짜와 일치하는 날씨 데이터 찾기
            activity_date = activity['date'].strftime('%Y%m%d')
            matching_weather = next((w for w in activity['weather'] if w['date'] == activity_date), None)
            
            if matching_weather:
                try:
                    sky_status_encoded = [0, 0, 0]
                    sky_status_encoded[['맑음', '구름 많음', '흐림'].index(matching_weather['skyStatus'])] = 1
                    
                    precip_type_encoded = [0, 0, 0, 0, 0]
                    precip_type_encoded[['없음', '비', '비/눈', '눈', '소나기'].index(matching_weather['precipitationType'])] = 1
                    
                    X_weather.append([
                        float(matching_weather['temperature']) - 5,  # temperature_min
                        float(matching_weather['temperature']) + 5,  # temperature_max
                        float(matching_weather['humidity']),
                        float(matching_weather['windSpeed']),
                        *sky_status_encoded,
                        *precip_type_encoded
                    ])
                    
                    X_climate.append([
                        float(climate['winter_temp']),
                        float(climate['summer_temp']),
                        float(climate['annual_precipitation']),
                        float(climate['winter_precipitation']),
                        float(climate['summer_precipitation']),
                        float(climate['snow_days']),
                        float(climate['foggy_days'])
                    ])
                    
                    processed_activity = preprocess_activity_name(activity['activityTag'])
                    y_new.append(activity_mapping[processed_activity])
                except (ValueError, KeyError) as e:
                    print(f"데이터 처리 중 오류 발생: {str(e)}. 해당 데이터를 건너뜁니다.")
                    continue
            else:
                print(f"활동 날짜 {activity_date}에 해당하는 날씨 데이터를 찾을 수 없습니다. 해당 데이터를 건너뜁니다.")
    
    return np.array(X_weather), np.array(X_climate), np.array(y_new)

def batch_learning():
    print("Starting batch learning...")
    global model, weather_scaler, climate_scaler
    
    # 현재 모델과 스케일러 로드
    model, weather_scaler, climate_scaler = load_trained_model()
    
    # 모든 사용자 활동 데이터 가져오기
    user_activities = get_user_activity_data()
    
    # 기존 데이터 로드
    climate_data, activity_data = load_data()
    
    X_weather, X_climate, y_new = preprocess_user_data(user_activities, climate_data, activity_data)

    if len(X_weather) > 0:
        # 데이터 정규화
        X_weather_normalized = weather_scaler.transform(X_weather)
        X_climate_normalized = climate_scaler.transform(X_climate)
    
        # 모델 재학습
        X_weather_tensor = torch.tensor(X_weather_normalized, dtype=torch.float32)
        X_climate_tensor = torch.tensor(X_climate_normalized, dtype=torch.float32)
        y_new_tensor = torch.tensor(y_new, dtype=torch.long)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(50):  # 에폭 수는 조정 가능
            optimizer.zero_grad()
            outputs = model(X_weather_tensor, X_climate_tensor)
            loss = criterion(outputs, y_new_tensor)
            loss.backward()
            optimizer.step()
        
        # 모델 저장
        save_model(model, weather_scaler, climate_scaler)
        
        # useractivities 컬렉션 비우기
        db = get_db()
        db['useractivities'].delete_many({})
        
        print(f"Batch learning completed. Model updated with {len(X_weather)} data points.")
    else:
        print("No data available for batch learning.")

def run_scheduler():
    schedule.every(7).days.do(batch_learning)
    while True:
        schedule.run_pending()
        time.sleep(3600)  # 1시간마다 체크

if __name__ == "__main__":
    climate_data, activity_data = load_data()
    X_weather, X_climate, y, weather_scaler, climate_scaler = preprocess_data(climate_data, activity_data)
    model = train_model(X_weather, X_climate, y)
    save_model(model, weather_scaler, climate_scaler, 'activity_recommendation_model.pth')
    
    # 스케줄러 실행 (백그라운드에서 실행되어야 함)
    import threading
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.start