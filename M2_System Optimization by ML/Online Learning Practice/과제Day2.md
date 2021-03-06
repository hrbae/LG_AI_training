# 온라인 학습을 활용한 시계열 예측

## 과제 내용
- 온라인 학습은 새로운 데이터에도 이전의 학습을 통해 유연하게 어느정도의 예측력을 보여주는 모델입니다. 이를 위해서라면 이전의 학습을 통해 파라미터 값을 어떻게 정하느냐가 중요한 목표가 됩니다. 과제는 크게 두 가지 입니다.
1) 실습파일 중 "code"라고 명시된 부분 작성하기(코딩)
2) Online learning practice 폴더에 'SCFI_data.csv' 데이터를 배치LSTM모델과 OnlineLSTM모델에 적용해보고 자신이 정한 파라미터(epoch, learning_rate)값과 그 이유를 설명하기

- 제출기한: 2022.06.24 금 18:00
- 제출처: 김도희 박사과정(kimdohee@pusan.ac.kr)
- 제출기한이 끝나면 참고하실 수 있도록 파일 업로드 예정이며 실습과 관련해 궁금하신 분들은 메일로 문의 부탁드립니다.

## 참고1. 데이터 설명
- SCFI 데이터는 Sanhai Containerized Freight Index로 상하이 컨테이너 운임지수 데이터입니다. 컨테이너선의 운임을 나타내는 대표 지표 중 하나이며 코로나19 발생 이후 급상승한 데이터입니다. 온라인 학습을 통해 급변동하고 새로 추가될 데이터에 대해 배치학습보다 잘 반응한다는 것을 보여줄 수 있는 데이터니 참고해주세요.

## 참고2. 데이터 로드
- pd.read_csv('https://github.com/hrbae/LG_AI_training/blob/main/M2_System%20Optimization%20by%20ML/Online%20Learning%20Practice/SCFI_data.csv')

![KakaoTalk_20220610_113020333](https://user-images.githubusercontent.com/58931222/172979071-277205b7-5953-4e7c-90b1-a528484ac4cf.png)
