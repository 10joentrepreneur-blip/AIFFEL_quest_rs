 # FUNDAMANTAL_03.

## 1. 프로젝트 정보
1.1. 제목: 3.Evaluation Metric  
1.2. 소개: **성능평가 척도((Evaluation Metric)**는 머신러닝 모델의 성능을 정량적으로 측정하는 기준  
1.2.1. 프로젝트 목적: 분류 모델의 성능 평가 지표를 중심으로 다양한 관점을 이해하고, 각 분야의 대표적인 평가 지표와 그 필요성을 이해한다.

1.2.2. 해결하려는 문제: 
	- 분류(Classification) 모델 평가 지표의 중요셩, 한계와 그이유   
	- 회귀(Regression) 모델 평가 지표의 계산
1.2.3. 사용된 주요 기술: Python_Colab.  
1.3. 참여자: 박정훈 / 학습을 이해하고 실행 및 기록.

---

## 2. 목차
Loss와 Metric
Loss와 Metric의 차이는 무엇일까요?
Confusion Matrix 와 Precision/Recall
Confusion Matrix 와 Precision/Recall의 개념에 대해 알아봅니다.
Threshold의 변화에 따른 모델 성능
Threshold를 변경하며 모델 성능을 확인하는 실습을 해봅니다.
Precision-Recall 커브
Precision-Recall 커브를 직접 코드로 그려보며 알아봅니다.
ROC 커브
ROC 커브에 대해 살펴봅니다.
다양한 머신러닝 모델의 평가척도
이외에도 몇 가지 평가척도에 대해 살펴봅니다.

---

## 3. 문제 정의 및 목표
3.1. 배경: 아이펠 리서치 15기 FUNDAMANTAL 기초 학습 중 성능평가 과정  
3.2. 목표: 
- 머신러닝 학습 결과를 평가할 수 있습니다.
- Precision과 Recall의 관계를 이해할 수 있습니다
- AUC 영역을 통해 모델 간의 퍼포먼스를 비교할 수 있습니다.

---

## 4. 데이터셋
4.1. (출처: 사이킷런 plot-precision-recall 예제)https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

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
6.1. 학습 과정:  
6.1.1. Loss와 Metric : 우리가 원하는 모델은 테스트 데이터에 대한 Accuracy가 높은 모델
	- Loss : 모델 학습시 학습데이터(train data) 를 바탕으로 계산되어, 모델의 파라미터 업데이트에 활용되는 함수
  - Metric : 모델 학습 종료 후 테스트데이터(test data) 를 바탕으로 계산되어, 학습된 모델의 성능을 평가하는데 활용되는 함수
	- 차이 
		Loss는 기계(모델)를 위한 것이고, Metric은 사람을 위한 것입니다.
		Loss는 학습 최적화에 필수적인 미분 가능하고 연속적인 피드백을 제공합니다.
		Metric은 모델 성능 평가를 위해 직관적이고 이해하기 쉬운 정보를 제공합니다

6.1.2. Confusion Matrix 와 Precision/Recall
	- 정확도(Accuracy)= TP+TN/TP+TN+FP+FN, (정답을맞힌예측의수)/(전체문제의수)
​	 - 평가 척도의 오류 점검 방법 
			True Positive (TP) - 양성(Positive)을 양성으로 맞혔을 때
			True Negative (TN) - 음성(Negative)을 음성으로 맞혔을 때
			False Positive (FP) - 음성(Negative)을 양성(Positive)으로 잘못 예측했을 때
			False Negative (FN) - 양성(Positive)을 음성(Negative)으로 잘못 예측했을 때
			
			정밀도(Precision)= TP/TP+FP (암환자로 예측한 비율)
			재현율(Recall)= TP/TP+FN (암환자/암환자를 정상으로 예측한 비율)
			양성 데이터 분포가 불균형할 경우 정확도는 평가에 좋은 척도가 아님.
			
	-	솔루션
			F1 score : 정밀도+재현율의 조화 평균 값의 척도
			Fβ =(1+β2)⋅ precision⋅recall/(β2⋅precision+recall)

	6.1.3. Threshold(분류기준:임계값)의 변화에 따른 모델 성능
		     전체적인 모델의 성능을 평가하는 두가지 방법
			1. PR(Precision and Recall) 커브
			2. ROC(Receiver Operating Characteristic) 커브

			- 학습 과정을 통해 알 수 있는 내용

			1. 데이터 특성이 모델 성능에 미치는 영향.
				- 특성의 중요성 : 어떤 특성(feature)을 사용하느냐에 따라 크게 좌우.
				- 과적합(Overffitting): 특성 수의 영향으로 낮은 성능을 보임.
			2. 모델 선택 및 하이퍼파라미터의 중요성
				- 커널 선택: SVM 모델의 경우, 어떤 커널을 사용하느냐에 따라 복잡도와 성능이 달라짐.
					(POLY: 과적합 가능,Linear: 덜 민감/과적합 위험이 적은 모델)
			3. 다양한 평가 지표의 활용
				- Accuracy의 한계: 데이터가 불균형한상황에서 정확도가 높아도 중요한 오류 포함.
				- Confusion Matrix: 예측/실제 클래스를 표로 정리하여 어떤 종류의 오류를 범했는지 시각적으로 파악.
				- Precision & Recall : 
					정밀도: 모델이 '맞다고' 예측한 것 중 실제 '맞는' 비율. (1,4,7) =%
					재현율: 실제 '맞는' 것 중 모델이 제대로 찾아낸 비율.   (1~10중 2,3,8)=%
				- 분류보고서(Classification Report): 클래스별로 정리 제공.
			4. 결정 임계값(Decision Threshold)조정
				- 확신도 기반 예측: SVM 모델의 decision_function은 단순한 답을 넘어, 예측에 대한 모델의 확신도를 연속적인 값으로 제공.
				- 정밀도-재현율 트레이드오프: 기본 임계값(0)을 조정하여 트레이드오프 관계를 직접 확인.목적에 따라 임계값을 조절하여 최적의 성능을 찾을수 있음.
			5. 머신러닝 워크플로우의 이해
				- 데이터준비->모델 선택 및 훈련->성능평가 = 사이킷런

	6.1.4. Precision-Recall 커브 : 클래스 불균형이 심한 데이터셋에서 더 정확하게 평가한다.
				- PR(Precision-Recall) 커브는 Recall을 X축, Precision을 Y축에 놓고 Threshold 변화에 따른 두 값의 변화를 그래프
				- AUC(Area Under Curve): Threshold 값에 무관하게 모델의 전체적인 성능을 평가하는 방법
					PR 커브 아래쪽 면적을 계산하는 방법 : average_precision_score 함수를 사용

			- 학습 과정을 통해 알 수 있는 내용

			1. 클래스 불균형 문제에 대한 최적화된 지표
				- PR 커브의 필요성 : 정확도가 높게 나오더라도 모델 성능을 제대로 평가하기 어려울 때 
				- 정밀도와 재현율의 관계: 임계값에 따라 변화를 시각적으로 보여줌.
			2. 모델 성능의 종합적인 평가
				- 그래프 해석: 그래프 오른쪽 위로 가고, 곡선아래 면적이 넓을수록 성능이 좋은 모델 
				- Average Precision(AP): 단일 값을 통해 성능 비교 쉬움, 1에 가까울수록 우수한 모델.
			3. 목적에 따른 최적 임계값 선택
				- 시각화 : 임계값에 따라 정밀도와 재현율의 트레이드오프 관계를 직관적으로 이해.
				- 임계값 조정 :
						- 정밀도가 중요할 때 : ex)스팸 메일:  잘못 예측 하는 것을 최소화해야 할 경우.
						- 재현율이 중요할 때 : ex)암 진단  :  놓치는 것을 최소화해야 할 경우.
			4. 모델의 예측 확신도 활용
				- decision_function의 역할 : PR 커브를 그릴 때 모델의 최종 예측값(predict) 대신, 모델의 확신도(decision_function)를 사용
				- 연속적인 정보 : 연속적인 값을 통해 임계값을 세밀하게 조정하며 정밀도와 재현율의 변화를 분석

		  결론
			1. 클래스 불균형 문제 해결(데이터 전처리)
			2. 모델 학습 및 개선: 모델 튜닝 방향성 설정 (하이퍼파라미터 최적화)
			3. 모델 선택 : 성능 종합 비교(svm,랜덤 포레스트등)/PR커브,AP정수 활용
			4. 목적에 맞는 최적 임계값 결정
​		
6.1.5. ROC 커브
			- ROC(Receiver Operating Characteristic Curve) 는 수신자 조작 특성 곡선
			  Confusion Matrix 수치를 활용해, 분류 능력을 그래프로 표현하는 방법

			- 학습 과정을 통해 알 수 있는 내용
			1. 모델의 분류 능력 비교: 
				- 모델 성능을 비교하는데 효과적인 시각적 도구.왼쪽 상단 모서리에 가까울수록 더 좋은 성능.
			2. AUC 값으로 성능 정량화 : 0.5~1까지의 범위, 값이 클수록 변별력 우수.
			3. 임계값에 따른 트레이드오프 : TPR,FPR의 변화 & 최적화
			4. 클래스 불균형에 대한 민감도 : 양성 클래스 예측 성능에 초점을 맞춘 PR 커브와 달리 클래스 불균형에 덜 민감.

			결론
			1. 커널 선택의 중요성 : 커널(linear, poly, rbf, sigmoid) 데이터 특성에 따라 최적의 커널이 달라짐.
														- 선형 커널(linear)과 비선형(poly,rbf) 커널의 활용성
			2. 모델의 일반화 능력 : ​ROC 커브와 AUC 값을 비교함으로, 어떤 커널 모델이 더 나은 일반화 성능을 보이는지 파악.
														- AUC 값이 높을수록 능력이 우수.
			3. 평가 지표의 종합적 활용 : ROC 커브와 AUC는 PR커브+AP 점수와 함께 사용될때 모델 성능에 더 최적화 그래프를 제공.
														- 클래스 불균형 문제에서는 PR커브와 ROC/AUC를 모두사용해서 모델이 양성 클래스를 얼마나 잘 예측 하는지 + 전반적인 분류 능력 모두를 평가하는 것을 추천.
			4. 모델 최적화의 지속성 : 
										
6.1.6. 다양한 머신러닝 모델의 평가척도
			- 머신러닝에는 분류만 있는 것이 아니라 회귀, 추천, 군집 등 다양한 모델이 있음
			- 음성, 이미지, 텍스트 등의 생성형 모델들이 가지는 독특한 평가 척도들도 매우 다양
			
			- 회귀 모델의 평가 척도 : 회귀 모델의 경우에는 어떤 loss가 적합한지 그 특성을 알아보는 것이 중요
			- 랭킹 모델의 평가 척도 : 추천시스템은 넓게 보면 정보 검색(Information Retrieval)과 같은 로직을 갖고 랭킹(우선순위) 개념이 추가됨.
			- 이미지 생성 모델의 평가 척도 : 정답과 모델의 예측치 사이의 거리를 측정하는 방식
			- 기계번역 모델의 평가척도 : BLEU (Bilingual Evaluation Understudy): 기계 번역 결과가 얼마나 좋은지를 평가

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
8.1. 결과 요약: 다양한 SVM 커널이 모델 성능에 미치는 영향을 PR/ROC 커브/AUC 지표 활용.  
8.2. 결과 시각화:각 그래프를 통한 직관적 비교 / 성능 진단.
---

## 9. 회고 및 개선 방향
9.1. 고민했던 점: 모르는 용어 / 코드 이해도 난해.  
9.2. 배운 점: 학습 프로젝트를 진행하면서 좀더 체계화 (학습중 README 동시 작성 & 연동을 위한 코드 모듈 작성화함)  
9.3. 향후 계획: 
			1. 익숙치 않은 구글 드라이브-코랩-깃허브 파일 연동 익히기. 
   		2. 4~5단계 하루에 마무리 해보기, 
			3. 8단계 마치고 블로그로 1단계 부터 다시 복습 겸 업로드 하기.

---

## 10. 참고 자료
10.1. 논문:  
10.2. 참고 웹사이트: Metrics and scoring https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics
			sklearn.metrics.average_precision_score https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
			SVM 참고 https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
				kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
			회귀의 오류 지표 알아보기 https://modulabs.co.kr/blog/regression_metrics_mae_mse
			정보 검색(Information Retrieval) 평가 방법: MAP, MRR, DCG, NDCG https://modulabs.co.kr/blog/information-retrieval-map-ndcg
			이미지 간 유사성 측정하는 방법 https://modulabs.co.kr/blog/how-to-measure-similarity




 