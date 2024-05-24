import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys

def clean_data(data):
    # Cabin 정보에서 덱 정보만 추출
    data['Deck'] = data['Cabin'].str.extract(r'([A-Za-z]+)')
    # 필요 없는 열을 삭제
    data.drop(['Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)
    # 범주형 데이터를 숫자로 변환
    data = pd.get_dummies(data, columns=['HomePlanet', 'Deck', 'Destination'], drop_first=True)
    return data

def main(train_path, test_path):
    # 학습 및 테스트 데이터를 불러오기
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # 데이터 클리닝
    train_data = clean_data(train_data)
    test_data = clean_data(test_data)

    # 특성과 타겟 변수를 분리
    X = train_data.drop('Transported', axis=1)
    y = train_data['Transported']

    # 학습 데이터를 훈련 세트와 검증 세트로 분리
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 랜덤 포레스트 모델을 초기화하고 학습
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 검증 세트에서 예측을 수행
    val_predictions = model.predict(X_val)

    # 검증 세트에서 정확도를 계산, 출력
    val_accuracy = accuracy_score(y_val, val_predictions)
    print("Validation Accuracy:", val_accuracy)

    # 테스트 데이터에 대한 예측을 수행
    test_predictions = model.predict(test_data)

    # 예측 결과를 CSV 파일로 저장
    submission = pd.DataFrame({'PassengerId': test_data.index, 'Transported': test_predictions})
    submission.to_csv('2011218.csv', index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python 2011218.py [train.csv path] [test.csv path]")
        sys.exit(1)
    
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    main(train_path, test_path)