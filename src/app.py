from flask import Flask, jsonify, request
from flask_caching import Cache
from model import (
    ActivityRecommendationModel, load_trained_model, load_data, get_weather_forecast_from_mongodb,
    run_scheduler, recommend_activity, recommend_activities_for_location, recommend_locations_for_activity,
    recommend_dates_for_activity_and_location, prepare_weather_input, prepare_climate_input, 
    preprocess_location_name, batch_learning, initial_batch_learning
)
from pymongo import MongoClient
import threading
from datetime import datetime
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
print("Current working directory:", os.getcwd())
print("Contents of current directory:", os.listdir())

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Get MONGO_URI from environment variables
mongo_uri = os.getenv('MONGO_URI')
app.config['MONGO_URI'] = mongo_uri

API_VERSION = 'v1'

# Logging setup
def setup_logging(app):
    handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

setup_logging(app)

# Load model and data
try:
    model, weather_scaler, climate_scaler = load_trained_model()
    print("Model loaded successfully")
except FileNotFoundError:
    print("Error: Pre-trained model not found. Please ensure the model file is included in the Docker image.")
    print(f"Current directory contents: {os.listdir()}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    sys.exit(1)


climate_data, activity_data = load_data()
weather_forecast = get_weather_forecast_from_mongodb()

# MongoDB client setup
client = MongoClient(mongo_uri)
db = client['actapp']
user_activities = db['useractivities']

# Start batch learning scheduler (run in background)
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

# 유효성 검사 함수
def validate_date(date_string):
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def validate_input(data, required_fields):
    for field in required_fields:
        if field not in data or not data[field]:
            return False, f"Missing required field: {field}"
    return True, ""

# 커스텀 에러 클래스
class APIError(Exception):
    def __init__(self, message, status_code=400, payload=None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

@app.errorhandler(APIError)
def handle_api_error(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.route(f'/api/{API_VERSION}/predict', methods=['POST'])
def predict_v1():
    try:
        app.logger.info(f"Received prediction request: {request.json}")
        data = request.json
        is_valid, error_message = validate_input(data, ['city', 'date'])
        if not is_valid:
            raise APIError(error_message, status_code=400)
        
        if not validate_date(data['date']):
            raise APIError('Invalid date format. Use YYYY-MM-DD', status_code=400)
        
        city = preprocess_location_name(data['city'])
        date = data['date']
        
        if city not in climate_data:
            raise APIError('Invalid city', status_code=400)
        
        if date not in weather_forecast[city]:
            raise APIError('Weather data not available for the specified date', status_code=400)
        
        weather = weather_forecast[city][date]
        recommended_activities = recommend_activity(model, weather, climate_data[city], weather_scaler, climate_scaler, activity_data)
        
        response = {
            'city': city,
            'date': date,
            'weather': weather,
            'recommended_activities': recommended_activities
        }
        
        app.logger.info(f"Prediction result: {response}")
        return jsonify(response)
    except APIError as e:
        raise
    except Exception as e:
        app.logger.error(f"Unexpected error in predict: {str(e)}")
        raise APIError(str(e), status_code=500)

@app.route(f'/api/{API_VERSION}/recommend/by_activity', methods=['POST'])
def recommend_by_activity_v1():
    try:
        app.logger.info(f"Received recommend by activity request: {request.json}")
        data = request.json
        is_valid, error_message = validate_input(data, ['activity'])
        if not is_valid:
            raise APIError(error_message, status_code=400)

        activity = data['activity']
        
        locations_with_scores = recommend_locations_for_activity(
            model, activity, climate_data, weather_forecast, 
            weather_scaler, climate_scaler, activity_data
        )
        
        recommended_dates = {}
        for location, _ in locations_with_scores:
            dates = recommend_dates_for_activity_and_location(
                model, activity, location, climate_data, weather_forecast,
                weather_scaler, climate_scaler, activity_data
            )
            recommended_dates[location] = [{'date': str(date), 'score': score} for date, score in dates]
        
        response = {
            'recommended_locations': [{'location': location, 'score': score} for location, score in locations_with_scores],
            'recommended_dates': recommended_dates
        }
        app.logger.info(f"Recommend by activity result: {response}")
        return jsonify(response)
    except APIError as e:
        raise
    except Exception as e:
        app.logger.error(f"Unexpected error in recommend by activity: {str(e)}")
        raise APIError(str(e), status_code=500)

@app.route(f'/api/{API_VERSION}/recommend/by_date', methods=['POST'])
def recommend_by_date_v1():
    try:
        app.logger.info(f"Received recommend by date request: {request.json}")
        data = request.json
        is_valid, error_message = validate_input(data, ['date'])
        if not is_valid:
            raise APIError(error_message, status_code=400)

        if not validate_date(data['date']):
            raise APIError('Invalid date format. Use YYYY-MM-DD', status_code=400)

        date = data['date']
        location, activity = recommend_location_and_activity(model, date, climate_data, weather_forecast, activity_data, weather_scaler, climate_scaler)
        
        response = {'recommended_location': location, 'recommended_activity': activity}
        app.logger.info(f"Recommend by date result: {response}")
        return jsonify(response)
    except APIError as e:
        raise
    except Exception as e:
        app.logger.error(f"Unexpected error in recommend by date: {str(e)}")
        raise APIError(str(e), status_code=500)

@app.route(f'/api/{API_VERSION}/recommend/by_location', methods=['POST'])
def recommend_by_location_v1():
    try:
        app.logger.info(f"Received recommend by location request: {request.json}")
        data = request.json
        is_valid, error_message = validate_input(data, ['location'])
        if not is_valid:
            raise APIError(error_message, status_code=400)

        location = preprocess_location_name(data['location'])
        
        activities_with_scores = recommend_activities_for_location(
            model, location, climate_data, weather_forecast, 
            weather_scaler, climate_scaler, activity_data
        )
        
        recommended_dates = {}
        for activity, _ in activities_with_scores:
            dates = recommend_dates_for_activity_and_location(
                model, activity, location, climate_data, weather_forecast,
                weather_scaler, climate_scaler, activity_data
            )
            recommended_dates[activity] = [{'date': str(date), 'score': score} for date, score in dates]
        
        response = {
            'recommended_activities': [{'activity': activity, 'score': score} for activity, score in activities_with_scores],
            'recommended_dates': recommended_dates
        }
        app.logger.info(f"Recommend by location result: {response}")
        return jsonify(response)
    except APIError as e:
        raise
    except Exception as e:
        app.logger.error(f"Unexpected error in recommend by location: {str(e)}")
        raise APIError(str(e), status_code=500)

@app.route(f'/api/{API_VERSION}/record_activity', methods=['POST'])
def record_activity_v1():
    try:
        app.logger.info(f"Received record activity request: {request.json}")
        data = request.json
        required_fields = ['location', 'activityTag', 'date', 'weather']
        is_valid, error_message = validate_input(data, required_fields)
        if not is_valid:
            raise APIError(error_message, status_code=400)
        
        if not validate_date(data['date']):
            raise APIError('Invalid date format. Use YYYY-MM-DD', status_code=400)
        
        data['date'] = datetime.strptime(data['date'], '%Y-%m-%d')
        
        user_activities.insert_one(data)
        
        response = {'message': 'Activity recorded successfully'}
        app.logger.info(f"Record activity result: {response}")
        return jsonify(response), 200
    except APIError as e:
        raise
    except Exception as e:
        app.logger.error(f"Unexpected error in record activity: {str(e)}")
        raise APIError(str(e), status_code=500)

@app.route(f'/api/{API_VERSION}/recommend/by_activity_and_location', methods=['POST'])
def recommend_by_activity_and_location_v1():
    try:
        app.logger.info(f"Received recommend by activity and location request: {request.json}")
        data = request.json
        is_valid, error_message = validate_input(data, ['activity', 'location'])
        if not is_valid:
            raise APIError(error_message, status_code=400)

        activity = data['activity']
        location = preprocess_location_name(data['location'])

        recommended_dates = recommend_dates_for_activity_and_location(
            model, activity, location, climate_data, weather_forecast, 
            weather_scaler, climate_scaler, activity_data
        )
        
        response = {'recommended_dates': [str(date) for date in recommended_dates]}
        app.logger.info(f"Recommend by activity and location result: {response}")
        return jsonify(response)
    except APIError as e:
        raise
    except Exception as e:
        app.logger.error(f"Unexpected error in recommend by activity and location: {str(e)}")
        raise APIError(str(e), status_code=500)

#-------------------------------------------------------------

@app.route(f'/api/{API_VERSION}/activities', methods=['GET'])
@cache.cached(timeout=3600)  # 1시간 동안 캐시
def get_activities_v1():
    app.logger.info("Received get activities request")
    activities = [activity['name'] for activity in activity_data]
    app.logger.info(f"Get activities result: {activities}")
    return jsonify(activities)

@app.route(f'/api/{API_VERSION}/cities', methods=['GET'])
@cache.cached(timeout=3600)  # 1시간 동안 캐시
def get_cities_v1():
    app.logger.info("Received get cities request")
    cities = list(climate_data.keys())
    app.logger.info(f"Get cities result: {cities}")
    return jsonify(cities)

if __name__ == '__main__':
    cache.clear()
    app.run(debug=True, host='0.0.0.0', port=5003)