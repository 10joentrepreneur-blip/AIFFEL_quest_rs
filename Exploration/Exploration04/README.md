# EXPLORATION 03

### 1.제목: 1.카메라 스티커앱 만들기 첫걸음

**1.1. 소개:**
*   **프로젝트 목적:**
*   **해결하려는 문제:**
					1.

*   **사용된 주요 기술:** Python (Google Colab 환경)



### 2. 학습 목차

*   **2.1. 1. 카메라 스티커앱 만들기 첫걸음   **
*   **2.2. 2. 어떻게 만들까? 사진 준비하기  **
*   **2.3. 3. 얼굴 검출 face detection  **
*   **2.4. 4. 얼굴 랜드마크 face landmark  **
*   **2.5. 5. 스티커 적용하기 **

### 3. 문제 정의 및 목표

**3.1. 배경:**
*

**3.2. 목표:**
*   ** 얼굴인식 카메라의 흐름을 이해합니다. **
*   ** dlib 라이브러리 사용하여 실습해봅니다.**
*   ** 이미지 배열의 인덱싱 예외 처리를 익힙니다.**

**3.3. 학습내용:**
*   ** 어떻게 만들까? 사진 준비하기: **OpenCV 라이브러리를 통해 실습을 준비
*   ** 얼굴 검출 face detection: **face detection 실습
*   ** 얼굴 랜드마크 face landmark: ** 이미지에 face landmark를 적용
*   ** 스티커 적용: ** 스티커 적용하기를 통해 어플의 초기 버전을 완성


### 4. 데이터셋
*   **OpenCV 라이브러리**

**4.1. 데이터셋 출처:**
*   **
**4.2. 데이터셋 설명:**
*   **:**

### 5. 기술 스택 및 라이브러리

**5.1. 주요 기술:**
*

**5.2. 요구 사항:**
*

### 6. 학습 과정
#.
**1.1. 카메라 스티커앱 만들기 첫걸음:**
	*   ** 코스 소개 **
1. 카메라앱 만들기를 통해 동영상 처리, 검출, 키포인트 추정, 추적, 카메라 원근의 기술을 다룹니다.
2. 간단한 스티커부터 시작해서 각도 변화가 가능하고 거리 변화에 강건한 스티커까지 만들 수 있습니다.

 *   ** 1-2. 어떻게 만들까? 사진 준비하기 **
1. 얼굴 포함 사진 준비
2. 얼굴 영역 bounding box를 찾고 face landmark 찾기.
3. 찾아진 영역으로 부터 머리에 왕관 스티커 붙여넣기

1-1. 얼굴 포함 사진 준비
1-2. 이미지 처리 관련 패키지 설치: opencv, cmake, dlib
1-3. 이미지 처리 출력을 위한 패키지 import
	import os # 환경 변수나 디렉터리, 파일 등의 OS 자원을 제어할 수 있게 해주는 모듈
	import cv2 # OpenCV라이브러리 → 컴퓨터 비전 관련 프로그래밍을 쉽게 할 수 있도록 도와주는 라이브러리
	import matplotlib.pyplot as plt # 다양한 데이터를 많은 방법으로 도식화 할 수 있도록 하는 라이브러리
	import numpy as np # 다차원 배열을 쉽게 처리하고 효율적으로 사용할 수 있도록 하는 라이브러리
	import dlib # 이미지 처리 및 기계 학습, 얼굴인식 등을 할 수 있는 c++ 로 개발된 고성능의 라이브러리
 * plt.imshow(img_rgb) // 기본 : 흑백 (0)

	*   ** 1-3. 얼굴 검출 face detection **
1. Object detection 기술을 이용해서 얼굴의 위치 찾기
	  (dlib의 face detector는 HOG(Histogram of Oriented Gradients)와 SVM(Support Vector Machine)을 사용해서 얼굴을 찾음)
		HOG는 이미지에서 색상의 변화량을 나타낸 것
		SVM은 선형 분류기-> 한 이미지를 다차원 공간의 한 벡터라고 보면 여러 이미지는 여러 벡터가 될텐데요. 이 여러 벡터를 잘 구분짓는 방법
		이미지가 HOG를 통해 벡터로 만들어진다면 SVM이 잘 작동

1-1. sliding window를 사용하여 얼굴 위치 찾기
		작은 영역(window)을 이동해가며 확인하는 방법: 큰 이미지의 작은 영역을 잘라 얼굴이 있는지 확인하고, 다시 작은 영역을 옆으로 옮겨 얼굴이 있는지 확인하는 방식
		이미지가 크면 클수록 오래걸리는 단점이 있습니다. 바로 이 지점이 딥러닝이 필요해지는 이유
		그럼 dlib을 활용해 hog detector를 선언

		detector_hog = dlib.get_frontal_face_detector() # 기본 얼굴 감지기를 반환

		detector_hog를 이용 얼굴의 bounding box를 추출
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		dlib_rects = detector_hog(img_rgb, 1)   # (image, num of image pyramid)

	 *  1= 파라미드 수 (Image Pyramids)
	 * ➰ upsampling이란,데이터의 크기를 키우는 것


	 	*   ** 1-4. 얼굴 랜드마크 face landmark **
		- 스티커를 섬세하게 적용하기 위해서는 이목구비의 위치를 아는 것이 중요
		- 이목구비의 위치를 추론하는 것을 face landmark localization 기술
		-  face landmark는 detection 의 결과물인 bounding box 로 잘라낸(crop) 얼굴 이미지를 이용

		Object keypoint estimation 알고리즘
		- 객체 내부의 점을 찾는 기술을 object keypoint estimation 이라고 함.
		1. top-down : bounding box를 찾고 box 내부의 keypoint를 예측
		2. bottom-up : 이미지 전체의 keypoint를 먼저 찾고 point 관계를 이용해 군집화 해서 box 생성

		Dlib landmark localization
		[Landmark 데이터셋 요약]
		AFLW dataset: Dlib은 ibug 300-W 데이터셋으로 학습한 pretrained model 을 제공

		*   ** 1-5. 5. 스티커 적용하기 **
		- 스티커 위치, 스티커 크기 고려



### 7. 실행 방법 (Google Colab)

**7.1. 환경 설정 및 파일 준비:**
1.  **코랩 노트북 사용**
2.  **구글드라이브 / 깃허브 연동**

**7.2. 실행:**
1.  **순차적 셀 실행**: 코랩 노트북의 코드 셀들을 학습 노드 순서에 따라 순차적으로 실행합니다.

### 8. 결과 및 시각화

**8.1. 결과 요약:**
*   **

**8.2. 결과 시각화:**
*   **모델 예측 결과 시각화:**:

### 9. 회고 및 개선 방향

**9.1. 고민했던 점:**

**9.2. 배운 점:**

**9.3. 향후 계획:**

### 10. 참고 자료

**10.1. 논문:**
*   (이번 학습에서는 참고한 논문이 없습니다.)

**10.2. 참고 웹사이트:**