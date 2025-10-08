 FUNDAMANTAL_01_2.
1. 프로젝트 정보

1.1. 제목: 2.다양한 데이터 전처리 기법
1.2. 소개:
1.2.1. 프로젝트 목적: 데이터 분석의 질 & 예측 모델의 성능을 높임
1.2.2. 해결하려는 문제: 분석결과의 신뢰도, 정확도 향상
(중복된 데이터,결측치,정규화,이상치,범주형 데이터,연속적인 데이터)
1.2.3. 사용된 주요 기술: Python_Colab.
1.3. 참여자: 박정훈 / 학습을 이해하고 실행 및 기록.

2. 목차

결측치(Missing Data)
중복된 데이터
이상치(Outlier)
정규화(Normalization)
원-핫 인코딩(One-Hot Endocing)
구간화(Binning)

3. 문제 정의 및 목표

3.1. 배경: 아이펠 리서치 15기 FUNDAMANTAL 기초 학습 중 데이터 전처리 과정
3.2. 목표:
목차의 학습 내용을 명확히 이해하고 실행 / 반복 학습

4. 데이터셋

4.1. trade.csv (관세청 수출입 무역 통계 에서 가공한 데이터)
4.2. vgsales.csv (캐글의 Video Game Sales 데이터셋)

5. 기술 스택 및 라이브러리

5.1. 주요 기술: Python
- 주요 라이브러리 :
Pandas(pd) : 데이터정리 : csv,excel등 파일 데이터 호출 / DataFrame(행과열로구성) 이라는 강력한 데이터 구조를 만든후 조작 및 분석
NumPy(np): 데이터 수치연산 :수학 연산(통계,선형대수)등 다차원 배열을 효율적으로 빠르게 수행
Matplotlib(plt): 데이터 시각화 : 다양한  그래프 생성 / 커스터마이징

- 사용되는 기술 :
1. 데이터 정제 : (Data Claeaning)
	- 오류를 식별하고 수정하여 신뢰성을 높임.
	( 결측치 / 이상치 / 중복 데이터 처리 )
2. 데이터 변환 : (Data Transformation)
	- 형태를 분석 목적에 맞게 변환
	( 정규화[min-max scaling], 표준화, 인코딩[원-핫,레이블], 바이닝 )
3. 데이터 통합 : (Data Intergration)
	- 여러 소스의 데이터를 결합하여 하나의 일관된 데이터 셋을 만듬
	( 데이터 결합 )
4. 데이터 축소 : (Data Reduction)
 	- 데이터의 크기를 줄여 저장 및 분석 효율을 높이는 과정
5. 특성 공학 : (Feature Engineering)
	- 기존 특성에 새로운 특성을 만드는 과정
 	(기존 데이터를 조합 하거나 변형하여 새로운 특성을 만듬)
5.2. 요구 사항: 프로젝트를 실행하기 위해 필요한 라이브러리 목록을
 trade.csv와 함께 안내합니다.

6. 학습

6.1. 학습 과정:

1. 주요 라이브러리 정의: 데이터 정리/연산/시각화  (pd,np,plt)

2. 결측치

	수집하는 과정에서 누락된데이터

- 제거 방법: 전체 데이터 호출 len(**), 결측치 개수 확인  -**.count(),
	  컬럼 전체가 결측치인 컬럼 삭제 **.trade.drop(‘**’,axis=1), 확인 **.head()
	  행 결측치 여부 확인  : **.isnull().any(axis=1), True만 추출 : **[
   **.isnull().any(axis=1)]
	  컬럼전부가 결측치인 행 삭제 : **.dropna(how=’all’,
  subset=[’컬럼명’,,,],inplace=True)
	  결측치 데이터 재 확인 : **[ **.isnull().any(axis=1)]

- 결측치 데이터 보완 / 대체
  - 특정값 지정: 같은 값으로 대체할 경우 데이터의 분산이 실제보다 축소,
  -. 평균,중앙값으로 대체 : 1과 동일한 현상
  -. 다른 데이터를 이용해 예측값 대체
  -. 시계열 특성의 데이터 경우 전후 데이터의 평균으로 대체
	**.loc[열,’컬럼’]=(**.loc[열,’컬럼’]+**.loc,’컬럼’]/2   결과: **.loc[[열]]
	무역수지=수출총액-수입총액
	**.loc[열,’컬럼’]=(**.loc[열,’컬럼’]-**.loc,’컬럼’])
	실습:
- 191번째 행(인덱스 191)의 '수출금액' 값을 저장
original_export_amount = trade.loc[191, '수출금액']
- 평균 계산에 사용할 데이터프레임의 슬라이스를 선택
- iloc을 사용해 186부터 191 이전까지 (앞 5개), 그리고 192부터 197 이전까지 (뒤 5개) 선택
before_five_rows = trade['수출금액'].iloc[186:191]
after_five_rows = trade['수출금액'].iloc[192:197]
- 앞 5개와 뒤 5개의 '수출금액' 데이터 합치기
all_rows_for_avg = pd.concat([before_five_rows, after_five_rows])
- 평균 계산
average_export_amount = all_rows_for_avg.mean()
- 191번째 행의 '수출금액' 값을 평균값으로 대체
trade.loc[191, '수출금액'] = average_export_amount
- 변경 사항 확인을 위해 185부터 195까지 다시 출력
print(trade.iloc[185:196])
- 원래 값과 변경된 값 비교
print("
--- 변경 사항 요약 ---")
print(f"원래 191번 행의 수출금액: {original_export_amount}") : 5291404.5
print(f"새로운 191번 행의 수출금액: {trade.loc[191, '수출금액']}") 6710503.1
print(f"계산된 평균값: {average_export_amount}") 6710503.1

3. 중복된 데이터

	제거 : 같은 값을 가진 데이터 없이 행별로 값이 유일해야 할때.
	중복된 행을 불리언시리즈로 전체 반환: **.duplicated()
	df 에서 중복된 행들만 필터링 하여 반환:  **[**.duplicated()]
	중복 행 조건 반환 : **[(**['컬럼']=='조건')&(**['컬럼']=='조건')]
	중복 삭제: **.drop_duplicates(inplace=True)
	업데이트된 중복 삭제:  df.drop_duplicates(subset=['컬럼'], keep='last', inplace =
True)

4. 이상치(Outlier)

 	정의 : 일반 값의 범위에서 벗어나 극단적으로 크거나 작은 값
	Min-Max Scaling 스케일링을 통해 입력된 데이터의 척도를 공정하게 기준
  	잡을때 이상치 값 하나가 전체 데이터 분포를 왜곡 시킬수 있음.
이상치 찾기 : anomaly detection
z score (평균을 빼주고 표준편차로 나눔): 데이터가 정규분포를 따를때 적합
가 정한 특정 기준을 넘어서는 데이터를 판단.(기준에 따라 이상치 데이터 양이 달라짐)
**.loc[outlier(**, '컬럼명', 2)]
평균165,표준편차(편차 제곱,분산) ±5일 경우 z score=2라면(5x2)=10 즉
155미만 또는 175를 초과 할 경우 이상치에 해당
IQR method (사분위 범위수): 데이터가 치우쳤거나 정규분포를 따르지 않을시 적합.
IQR=Q3-Q1(평균), Q1(하한),Q3(상한)
하한선 Q1-1.5 X IQR, 상한선 Q3+1.5 X IQR
제거 방법 :
outlier()함수와 ~ 연산자 사용
def outlier(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
이상치 행을 true로 반환 >
 return (df[col] < lower_bound) | (df[col] > upper_bound)
이상치 제거 >
- '**' 열의 이상치 마스크 생성
outlier_mask = outlier(**, '컬럼명')
- 이상치가 아닌(~) 행만 선택하여 새로운 데이터프레임 생성
**_cleaned = **[~outlier_mask]
- 결과 확인
print("이상치 제거 후 데이터 건수:", len(**_cleaned))
drop() 메서드 사용(인덱스 활용)
- 이상치인 행의 인덱스 찾기
- z=2를 기준으로 이상치를 판단
z_threshold = 2
- outlier() 함수에 z 인자를 전달
outlier_indices = **.loc[outlier(**, '컬럼명', z_threshold)].index
- 해당 인덱스를 가진 행을 제거 (원본에 바로 적용하지 않으려면 inplace=True 생략)
trade_cleaned = trade.drop(outlier_indices)
- 결과 확인
print("이상치 제거 후 데이터 건수:", len(trade_cleaned))
대체 방법 :
1. 평균 또는 중앙값으로 대체 : (median_value)
	적용: 데이터 분포 안정적이고 이상치가 적을때
2. 윈저화(Winsorizing) 또는 capping: (IQR)
	적용 : 이상치가 많고 분포가 치우쳐 있을때
3. 보간법(Interpolation) : interpolate()
	(이상치 주변의 데이터 포인트로 추정 / 대체)
	적용 : 시계열 데이터에서 추세가 중요할때

5. 정규화 (Normalizaition)

데이터의 스케일을 조정하여 일정한 범위에 맞추는 포괄적인 개념
	적용: 컬럼 간에 범위가 크게 다를 경우 (가우시안 분포)

표준화(Standardization) : 평균을 0, 표준편차(분산) 1로 변환
	Z-score
		# 정규분포를 따라 랜덤하게 데이터 x를 생성합니다.
x = pd.DataFrame({'A': np.random.randn(100)*4+4,
                 'B': np.random.randn(100)-1})
x
- 데이터 x를 Standardization 기법으로 정규화합니다.
x_standardization = (x-x.mean())/x.std()
x_standardization
	standard scaling (평균을 0으로 만드는 과정)
Z=(𝑥-𝜇)/𝜎
	Z=(각 데이터 값 - 데이터의 평균) / 표준편차
	표준편차 공식
	1. 평균 구하기
		𝜇 = ∑𝑥i / N (평균 =  모든 데이터 값의 합 / 총 개수)
2. 편차 구하기
		𝑥i - 𝜇           (각 데이터 값 - 평균)
	3. 편차 제곱하기
	(𝑥i - 𝜇)²        (각 데이터 값 - 평균)* 제곱
	4. 분산 구하기
		제곱한 편차들의 합 / 데이터의 총 개수
	5. 표준편차 구하기
	 𝜎=√∑(𝑥i /  𝜇)²/N

Min-Max Scaling : 최솟값은 0, 최댓값은 1로 변환, 이상치에 매우 민감
- 데이터 x를 min-max scaling 기법으로 정규화합니다.
x_min_max = (x-x.min())/(x.max()-x.min())
x_min_max

Robust Scaling : (IQR:사분위 범위) 이상치에 덜 민감하게 데이터를 변환

	6. 원-핫 인코딩

  문자열로 된 범주형 데이터를 머신러닝 모델이 이해할 수 있는 숫자
1, 0 형태로 변환하는 전처리 기법
country = pd.get_dummies(trade['국가명'])
국가명 컬럼을 원-핫 인코딩한 결과를 country라는 새로운 프레임에 저장

	범주에 속하면 1, 속하지 않으면 0 또는 True/False로 표시

7. 구간화(binning)

	연속적인 데이터를 구간을 나눠 분석할 때 사용하는 방법
	ctg = pd.cut(데이터명(), bins=구간 데이터 설정)
	print('데이터명[0]:', 데이터명[0])
print('데이터명[0]가 속한 카테고리:', ctg[0])
-ctg 범주에 속한 데이터명 값 print
	ctg.value_counts().sort_index() 구간별 총 개수 확인
	- 6을 입력해서 cut() 최솟값부터 최대값까지의 전체 범위를 6개의 동일한
크기의 구간으로 나눔
ctg = pd.cut(salary, bins = 6)
- 구간별로 값이 몇 개가 있는지 확인
ctg.value_counts().sort_index()
	- 데이터의 분포를 비슷한 크기의 그룹으로 나눔
	ctg = pd.qcut(salary, q=5)
- 구간별로 값이 몇 개가 있는지 확인
	print(ctg.value_counts().sort_index())

7. 실행 방법

7.1. 환경설정:
1. 구글 코랩에서 프로젝트 노트북 파일 오픈
2. 2.다양한 데이터 전처리 기법.ipynb 코드 작성
3. 필요 라이브러리 설치
4. 데이터 파일 준비
5. 구글 드라이브 연동
6. 깃 허브 연동
7.2. 실행:
1. 코랩 노트북의 코드 셀들을 학습 노드 순서대로 실행
2. 실행중 모르는 코드 / 함수 및 에러생길 시  구글 ai 모드에서 Q&A를 통해 에러 확인 및 수정 하여 실행.

8. 결과 및 시각화

8.1. 결과 요약:
코랩의 파이선 코드 / 라이브러리를 통해 데이터 전처리 기법을 이해하고
다양한 방법을 실행해보면서 전처리에 대한 인사이트를 얻음.
8.2. 결과 시각화:
제거 / 대체 / 평균의 공정성을 위한 척도 기준 생성 / 스케일링을 통한 최적화,
범용 데이터 처리 / 구간화를 통한 그룹 형성

9. 회고 및 개선 방향

9.1. 고민했던 점: 모르는 용어 / 코드 이해도 난해.
9.2. 배운 점: 이번 프로젝트를 통해 학습중 이해도에 따라 시간이 예상 보다 오래 약 3일 정도 걸릴 수 도 있다는 것을 배움, 학습중 모르는 것이 많고 실습을 하면서 에러 발생하는 부분이 발생하면서 (실행중 데이터를 제거 해버림으로 실습 결과물이 달라짐) 대처 하는데 시간이 소요됨으로 실습 실행시 이부분을 고려 해야 함을 배움.
9.3. 향후 계획:
1. 노드 학습 중 readme 파일 병행 작성.
2. 익숙치 않은 구글 드라이브-코랩-깃허브 파일 연동 익히기.
3. 학습 자료를 깃 허브로 관리하고 블로그로 학습 내용을 공유.

10. 참고 자료

10.1. 논문:
10.2. 참고 웹사이트:
https://youtu.be/FDCfw-YqWTE 정규화 영상 