 # FUNDAMANTAL_04.

## 1. 4.사이킷런으로 구현해 보는 머신러닝
1.1. 소개: 머신러닝의 다양한 알고리즘
1.2. 프로젝트 목적: 다양한 알고리즘에 대한 이해
1.3. 해결하려는 문제:
		-
		-
1.4. 사용된 주요 기술: Python_Colab.
1.5. 참여자: 박정훈 / 학습을 이해하고 실행 및 기록.

---

## 2. 목차
머신러닝 알고리즘
사이킷런에서 가이드하는 머신러닝 알고리즘
Hello scikit-learn
사이킷런의 주요 모듈 1. 데이터 표현법
사이킷런의 주요 모듈 2. 회귀 모델 실습
사이킷런의 주요 모듈 3. datasets 모듈
사이킷런의 주요 모듈 4. 사이킷런 데이터 셋을 이용한 분류 문제 실습
사이킷런의 주요 모듈 5. Estimator
훈련 데이터와 테스트 데이터 분리하기

---

## 3. 문제 정의 및 목표
3.1. 배경: 아이펠 리서치 15기 FUNDAMANTAL 기초 학습 중 성능평가 과정
3.2. 목표:
- 머신러닝 다양한 알고리즘을 이해
- 사이킷런 라이브러리의 사용법 익힘
- 사이킷런에서 데이터를 표현하는 방법을 이해,훈련용 데이터셋과 테스트용셋으로 데이터를 나누는 방법을 이해

---

## 4. 데이터셋
4.1.


---

## 5. 기술 스택 및 라이브러리
5.1. 주요 기술: Python

- 주요 라이브러리 :
  Pandas(pd) : 데이터정리 : csv,excel등 파일 데이터 호출 / DataFrame(행과열로구성) 이라는 강력한 데이터 구조를 만든후 조작 및 분석
  NumPy(np): 데이터 수치연산 :수학 연산(통계,선형대수)등 다차원 배열을 효율적으로 빠르게 수행
  Matplotlib(plt): 데이터 시각화 : 다양한 그래프 생성 / 커스터마이징

- 사용되는 기술 :
  1. 데이터 정제 : (Data Claeaning) - 오류를 식별하고 수정하여 신뢰성을 높임. ( 결측치 / 이상치 / 중복 데이터 처리 )
  2. 데이터 변환 : (Data Transformation) - 형태를 분석 목적에 맞게 변환 ( 정규화[min-max scaling], 표준화, 인코딩[원-핫,레이블], 바이닝 )
  3. 데이터 통합 : (Data Intergration) - 여러 소스의 데이터를 결합하여 하나의 일관된 데이터 셋을 만듬 ( 데이터 결합 )
  4. 데이터 축소 : (Data Reduction) - 데이터의 크기를 줄여 저장 및 분석 효율을 높이는 과정
  5. 특성 공학 : (Feature Engineering) - 기존 특성에 새로운 특성을 만드는 과정 (기존 데이터를 조합 하거나 변형하여 새로운 특성을 만듬)

5.2. 요구 사항: 프로젝트를 실행하기 위해 필요한 라이브러리 목록을 trade.csv와 함께 안내합니다.

---

## 6. 학습
6.1. 머신러닝 알고리즘
				- 3분류 / 대표적 알고리즘
				-  알고리즘을 분류 기준
							1. 라벨 유무- 유- 지도학습, 무-비지도학습
							2. 유- 지도학습 : 라벨이 연속이면 회귀, 범주형이면 분류
								 무- 비지도학습 : 클러스터링,차원축소
							3. 데이터 양에 따라 대용량 데이터에 적합한 알고리즘인지
							4. 데이터의 종류/특성에 따라 선형or 비선형 모델 사용할지 결정.

					1. 지도학습 (Supervised Learning)
							- 분류(Classification): 범주형 변수일 때
							- 회귀(Regression): 연속적인 값일 때
							- 예측(Forecasting): 과거 및 현재 데이터를 기반으로 미래를 예측할 때
					2. 비지도학습 (Unsupervised Learning)
							- 클러스터링(Clustering) : 비슷한 특성을 가진 데이터들을 그룹으로 묶는 작업
							- 차원 축소(Dimensionality Reduction) : 데이터의 특성(feature) 수를 줄여 데이터를 더 간단하게 만드는 작업
							- 연관 규칙 학습(Association Rule Learning) : 데이터 항목들 간의 숨겨진 관계를 찾음
					3. 강화학습 (Reinforcement Learning)
							- Monte Carlo methods
							- Q-Learning
							- Policy Gradient methods
						    기본 용어
								 - 에이전트 : 학습 주체 시스템
								 - 환경     : 에이전트에게 주어진 환경, 상황, 조건
								 - 행동     : 환경으로부터 주어진 정보를 바탕으로 에이전트가 판단한 행동
								 - 보상     : 행동에 대한 보상을 머신러닝 엔지니어가 설계

6.2. 사이킷런에서 가이드하는 머신러닝 알고리즘
				- 4가지 알고리즘의 Task
							1. 분류(Classification)
									- 6개
										선형 모델: SGDClassifier, LinearSVC
										비선형 모델: KNeighborsClassifier, SVC(커널 사용)
										확률 기반 모델: NaiveBayes
										앙상블 모델: EnsembleClassifiers
							2. 회귀(Regression)
									- 9개
										선형 모델: LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
										트리 기반 모델: Decision Tree Regressor, Ensemble Regressors
										서포트 벡터 머신 기반 모델: SVR
										최근접 이웃 기반 모델:KNeighborsRegressor
							3. 클러스터링(Clustering)
							4. 차원 축소(Dimensionality Reduction)

6.3. Hello scikit-learn
					- 사이킷런의 핵심은 바로 '표준화'와 '조립식 구조'입니다. 이 덕분에 복잡한 머신러닝 모델 개발을 누구나 쉽게 배우고, 효율적으로 자동화할 수 있게 됩니다.
						1. 일관성 있는 API 디자인: 어떤 머신러닝 모델(회귀, 분류 등)을 사용하든,
																			 fit()과 predict()라는 동일한 메서드를 사용
						2. 표준화된 워크플로우 : Transformer로 데이터 전처리, Estimator로 모델 학습,
																		 predict 메서드로 예측하는 표준화된 순서
						3. 파이프라인(Pipeline)을 통한 자동화: Pipeline: StandardScaler → SVC처럼
																		 여러 단계의 복잡한 워크플로우를 연결하고 자동화
						4. 접근성과 확장성 : 사이킷런의 규칙(Estimator 또는 Transformer의 API)만 따른다면,
																 개발자가 직접 만든 전처리 기능이나 모델도 파이프라인에 쉽게 추가

6.4. 사이킷런의 주요 모듈 (1) 데이터 표현법

					1. 사이킷런의 알고리즘은 파이썬 클래스로 구현
					2. 데이터셋은 NumPy의 ndarray, Pandas의 DataFrame, SciPy의 Sparse Matrix를 이용
					3. 훈련과 예측 등 머신러닝 모델을 다룰 때는
						CoreAPI라고 불리는 fit(), transfomer(), predict()과 같은 함수들을 이용

					4. 데이터 표현 방식 (2가지)
						1.특성행렬(Feature Matrix): 입력 데이터를 의미.
							- 특성(열:n_features:열의개수), 표본(행:n_samples:행의개수), 변수명 X
								[n_samples, n_features]은 [행, 열] 형태의 2차원 배열 구조를 사용
						2.타겟 벡터(Target Vector): 입력 데이터의 라벨을 의미.
							- 목표, n_samples:라벨의 개수, 변수명 y, 보통 1차원 벡터

6.5. 사이킷런의 주요 모듈 (2) 회귀 모델 실습

					- 데이터 생성 및 분석
					1. 생성: numpy를 사용하여 y = 2x - 3 * random 노이즈 포함 가상데이터
					2. 분석: 데이터가 선형적 패턴이지만 노이즈로 인해 흩어져 있는 것을 산점도로 시각화.
					- 데이터 형태 변환
					1. 넘파이 : reshape(): train_test_split이나 fit과 같은 사이킷런 함수는
											입력 데이터로 1차원 배열을 2차원 배열로 변환

				  - 모델 학습 및 예측
					1. 모델 생성 : LinearRegression 선형 회귀 모델
					2. 파이프라인 활용: Pipeline을 통해 StandardScaler와 LinearRegression을 연결하여,
						 데이터 전처리(스케일링)부터 모델 학습까지의 과정을 자동화
					3. 예측 : np.linspace()를 통해 생성된 새로운 데이터에 대한 예측을 수행

				  - 모델 평가
					1. RMSE 계산: sklearn.metrics.mean_squared_error를 사용하여 모델의 평균 제곱 오차(MSE)를 구하고,
					   np.sqrt()를 통해 평균 제곱근 오차(RMSE)를 계산
					2. 모델 적합성 시각화:산점도와 회귀선을 함께 시각화하여,
					   학습된 모델이 데이터의 패턴을 얼마나 잘 반영하는지 시각적으로 확인

6.6. 사이킷런의 주요 모듈 (3) datasets 모듈
					- dataset loaders
					- dataset fetchers
						- Toy dataset
							-datasets.load_wine()
					1. data(특성행렬,shape_배열확인,ndim_차원확인)
					2. target(타겟 벡터,shape_배열확인)
					3. feature_names_(특성이름,len()_데이터/특성행령 개수확인)
					4. target_names_(분류)
					5. descr(데이터 설명)

6.7. 사이킷런의 주요 모듈 (4) 사이킷런 데이터 셋을 이용한 분류 문제 실습
					1. 특성행렬 Pandas DataFrame으로 나타내기
					2. 머신러닝
						- 모델생성: model = RandomForestClassifier()
						- 훈련 : model.fit(X, y)
						- 예측 : model.predict(X)
						- 성능평가 : print(classification_report(y, y_pred))
												 print("accuracy = ", accuracy_score(y, y_pred))

6.8. 사이킷런의 주요 모듈 (5) Estimator
					- 모델 훈련에서 핵심이 되는 API
					1.Estimator 객체 : 데이터셋을 기반으로 머신러닝 모델의 파라미터를 추정하는 객체.
														 사이킷런의 모든 머신러닝 모델은 Estimatorm라는 파이썬 클래스로 구현.
						- 추정하는 과정=훈련 : Estimator fit()메서드
						- 예측               : Estimator predict()메서드

				ex) 와인 분류 문제를 해결하는 과정 

				model=RandomForestClassifier()--------------Model(Estimator)
																																l
												Traigning Data(Feature Matrix)-l				l
				model.fit(X,y)---------------------------------l-----> fit
											  Training Label(Target Vector) -l        l
																																l
				y_pred=model.predict(X)--Test Data(Feature Matrix)--Prediction

				ex) 선형 회귀 문제를 해결한다면 
				model=LinearRegression()==== 클래스를 사용

				ex) 타겟 벡터가 없는 데이터인 비지도 학습의 경우는 
						제외 --> Training Label(Target Vector)

6.9. 훈련 데이터와 테스트 데이터 분리하기
				- 동일한 데이터 훈련과 예측으로 인한 정확도 100% 의 오류 방지
				- 보통 훈련 데이터와 테스트 데이터의 비율은 8:2로 설정
				- 사이킷런에서는 이 필수 기능을 당연히 API로 제공
				-  model_selection의 train_test_split() 함수
				- X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	      1. 데이터셋 로드
				2. 훈련용 데이터셋 나누기<---
				3. 훈련하기 fit()
				4. 예측하기 predict()
				5. 정답률 출력하기 accuracy_score()			

---

## 7. 실행 방법
7.1. 환경설정:
1. 구글 코랩에서 프로젝트 노트북 파일 오픈
2. 3.Evaluation Metric.ipynb 코드 작성
3. 필요 라이브러리 설치
4. 데이터 파일 준비
5. 구글 드라이브 연동
6. 깃 허브 연동

7.2. 실행:
1. 코랩 노트북의 코드 셀들을 학습 노드 순서대로 실행
2. 실행중 모르는 코드 / 함수 및 에러생길 시 구글 ai 모드에서 Q&A를 통해 에러 확인 및 수정 하여 실행.

---

## 8. 결과 및 시각화
8.1. 결과 요약: 
								1. 분류(Classification) 모델:데이터셋을 사용하여 RandomForestClassifier 모델을 학습하고 평가  
									- train_test_split()을 사용해 훈련 데이터와 
									  테스트 데이터를 분리함으로써, 모델의 일반화 성능을 객관적으로 평가
								2. 회귀(Regression) 모델: y = 2x - 3 * random 형태의 가상 데이터를 사용해
								   LinearRegression 모델을 학습
									- RMSE 평가: mean_squared_error를 통해 모델의 예측 오차를 측정하고, 
									  오차의 크기를 원래 데이터의 단위로 해석하는 RMSE의 의미를 이해

8.2. 결과 시각화:각 그래프를 통한 직관적 비교 / 성능 진단.
								1. 회귀 분석 시각화 : 산점도와 회귀선을 함께 표시하여, 데이터의 분포와 모델이 
								   예측한 최적의 추세선을 한눈에 비교
								2. 파이프라인 구조 시각화 : set_config(display='diagram')을 활용하여 Transformer와 
								   Estimator가 연결된 파이프라인의 구조를 그림으로 확인
								
---

## 9. 회고 및 개선 방향
9.1. 고민했던 점: 
				1.개념의 추상성: Estimator, Transformer, Pipeline과 같은 용어들이 추상적으로 느껴져, 
					실제 코드와 어떻게 연결되는지 이해하는 데 어려움을 겪음
				2.워크플로우의 순서: 데이터를 준비하고 전처리한 뒤 모델을 학습하는 일련의 과정이 한 번에 와닿지 않았으며,. 
				  특히 train_test_split의 필요성과 역할에 대한 명확한 이해가 부족했음
				3.Pipeline의 작동 방식: Pipeline이 전처리를 '자동으로' 해준다는 것과 전처리 로직을 '설계'해야 하는 것
				  사이의 차이가 모호했음

9.2. 배운 점: 
				1. 사이킷런의 핵심 구조: Estimator와 Transformer라는 일관된 API를 통해 다양한 모델과 전처리기를
					 통일된 방식으로 다룰 수 있음.
				2. Pipeline의 효율성: 여러 단계를 묶어 코드를 간결하게 만들고, 데이터 누수와 같은 문제를 효과적으로 
				   방지하는 Pipeline의 자동화를 체감
				3. 평가의 중요성: 훈련 데이터와 테스트 데이터 분리는 객관적으로 평가를 하는 기준.
				4. 라이브러리-모듈-클래스-객체-메서드-파라미터의 순서
				5. 모르는 내용이 많아 학습량이 많다보니 하루에 2단계는 사실상 무리. 하루에 1단계 학습 가능함.


9.3. 향후 계획:
			1. 익숙치 않은 용어, 개념 학습
   		2. 하루 1단계 학습을 제대로
			3. 8단계 마치고 블로그로 1단계 부터 다시 복습 겸 업로드 하기.

---

## 10. 참고 자료
10.1. 논문:
10.2. 참고 웹사이트:
			1. 적절한 머신러닝 알고리즘을 선택하는 방법 https://modulabs.co.kr/blog/choosing-right-machine-learning-algorithm
			2. Reinforcement Learning KR https://github.com/reinforcement-learning-kr
			3. aikorea/awesome-rl https://github.com/aikorea/awesome-rl
			4, scikit-learn: API Reference https://scikit-learn.org/stable/api/index.html
      5. scikit-learn: Dataset loading utilities https://scikit-learn.org/stable/datasets
			6. scikit-learn: sklearn.utils.Bunch https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html#sklearn.utils.Bunch


 