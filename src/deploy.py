from src.model import initial_batch_learning, load_data, get_weather_forecast_from_mongodb, get_db
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
mongo_uri = os.getenv('MONGO_URI')
client = MongoClient(mongo_uri)
db = client['actapp']

def prepare_for_deployment():
    print("배포를 위한 모델 준비를 시작합니다...")
    
    try:
        # 초기 배치 학습 실행
        model, weather_scaler, climate_scaler, learning_success = initial_batch_learning()
        
        if learning_success:
            # 기타 필요한 데이터 로드
            climate_data, activity_data = load_data()
            
            # useractivities 컬렉션 비우기
            db = get_db()
            db['useractivities'].delete_many({})
            
            print("모델과 데이터 준비가 완료되었습니다. 배포 준비가 되었습니다.")
            print("useractivities 컬렉션이 비워졌습니다.")
        else:
            print("배치 학습이 진행되지 않았습니다. 기존 데이터는 유지됩니다.")
            print("모델 준비가 완료되었습니다. 배포 준비가 되었습니다.")
    except Exception as e:
        print(f"배포 준비 중 오류가 발생했습니다: {str(e)}")


if __name__ == "__main__":
    prepare_for_deployment()