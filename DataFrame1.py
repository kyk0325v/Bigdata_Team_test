import matplotlib
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import numpy as np
import io
import csv

# 판다스의 장점, 엑셀과 차이점
# 1.대용량 데이터(GB 단위 이상)를 다룰 수 있습니다. 엑셀은 데이터 용량이 100MB을 넘어가거나,데이터가 100만 행이 넘어가면 정상적으로 작동하지 않는 현상을 겪기도 합니다.
# 2. 복잡하나 처리 작업들을 비교적 손쉽게 할 수 있습니다.
# 3. 손쉽게 데이터를 결합하고 분리할 수 있습니다. 데이터를 합치고 관계 연산을 수행할 수 있습니다.
#
# 데이터 프레임의 구조
# .index : 각각의 행 데이터의 의미(이름)를 나타냅니다. 별도로 인덱스를 지정하지 않는다.
# .columns: 각각의 열 데이터의 의미(이름)를 나타냅니다.
# .values: 데이터 자체에 해당하면 값(value)를 나타냅니다.

# (절대경로파일)
f=open("C:/Users/3/PycharmProjects/pythonProject2/datasample.csv",'r',encoding='euc-kr')

# 파일 오픈 객체 f 생성
# 파일 오픈 open
# data1.csv파일 오픈
# read 모드
# 인코딩 euc-kr
# EUC-KR
# 한글을 2byte(16bit)로 변환하는 방법이다.
# 8비트 1바이트

# 문자마다 2byte의 코드값이 정해져있다.
# EUC-KR 코드표에는 한글 뿐만 아니라 숫자, 특수기호, 영문, 한문, 일어가 정의되어 있다.
# 그 외에 다른 문자를 사용하는 나라의 언어는 인코딩이 불가능하다.
# 인코딩 압축명 png jpg 디코딩

rdr = csv.reader(f) # 파일오픈 f객체를 csv의 reader함수로 읽고 그 결과를 rdr에 저장

for line in rdr:
	print(line)
# CSV를 읽어 온 rdr의 모든 행을 출력한다.

data = pd.read_csv("datasample.csv",encoding='euc-kr')

# 판다스의 read_csv함수로 raw data 불러와서 data 객체에 저장.
print("\n"*3)

###################시리즈 생성해보기
series1 = pd.Series([1,2,3,4])
print(series1)
print("\n"*3)
# 시리즈를 만들어 출력하면 인덱스와 같이 출력됨.
# 인덱스를 지정하여 출력도 가능
series1=pd.Series([100,200,300,400], index=['첫','둘','셋','넷'])
print(series1)
print("\n"*3)
# #인덱스로 접근가능.
print(series1['셋'])
print("\n"*3)
# 인덱스 여러개 접근 가능
# print(series1['첫','둘'])
print("\n"*3)

# 값들만 출력. numpy의 array로 반환 됨
# print(serires1.values)
print(series1.values)
print(type(series1))
print(type(series1.values))
print("\n"*3)

# print("numpy 배열 만들기")
# print("\n"*3)
#
# import numpy as np
# X=np.array([1,2,3,4])  # array 배열 만드는 함수
# # 리스트 1234가 numpy 배열형태로 변환되어 arrayx에 저장
#
# # 배열array란 - 기초적인 기본 자료형, 자료의 구조, 같은 타입의 데이터들의 모음, 순차적 저장, 인덱스 활용가능
# # 인덱스로 상대위치 파악 가능
# # 파이썬에는 배열 자료형이 존재하지 않음
# #
# # 배열이 리스트와 다른점
# # 리스트는 어떤 요소 타입이든 저장할 수있다
# # 리스트는 여러 타입 데이터를 담을 수 있다
# #
# # 벡터 배열 행렬
# # 배열 컴퓨터에서 사용하는 개념으로 수를 포하핳ㄴ
# # 벡터 1차원으로 묶은 수
# # 행벡터: 벡터 행으로
# # 열벡터: 열로만
# # 행렬 2차원으로 묶은 수
#
#
#
# print(X)
# print(type(x))
# array(1,2,3,4]
#
# ############2차원 배열 생성
# arrayxy = np.array([[1,2],[3,4]])
# print(arrayxy)
# print(type(arrayxy))
# print("\n"*3)
#
# ################# range => list => array 로 변환
# sample_range=range(10)
# # int()
# # list()
# # str()
# sample_list=listrange(sample_range)
# print(sample_list)
# print("sample_list의 자료형:",type(sample_list))
# sample_array= np.array(sample_list)
# print(sample_array)
# print("sample_array의 자료형:",type(sample_array))
#
# sample_list2=list[[1,2],[3,4]]
# sample_array2=np.array(sample_list2)
# print(sample_array2)
# print("sample_array의 자료형:",type(sample_array2))
#
# i = [1,2,3]
# j = [4,5,6]
#
# array_ij=np.array([i,j])
# print(array_ij)
# #([i,h]) i: 첫번째 행에 들어간 리스 j: 두번째 행에 들어간 리스트
#
# arange_array= np.arange(10)
# print(arange_array)
# arange_array=np.array(2,8)
# print(arange_array)
# arange_array=np.array(3,7,2) #start / end / step
# print(arange_array)
# arange_array=np.array(3,8,0.1) #0.1step 배열 생성
# print(arange_array)
#
# test_array1=np.linspace(1.0,5.0,num=5)
# print(test_array1)
# print(type(test_array1))
#
# test_array2=np.linspace(1.0,5.0,num=5,endpoint=False)
# #endpoint = False => End로 지정된 5.0을 포함하지 않는다.
# print(test_array2)
# print(type(test_array2))
#
# test_array3 =np.ones(5) # 5칸을 1로 다 채운 배열을 생성한다.
# print(test_array3)
# print(type(test_array3))
#
# test_array4 =np.zeros(5) # 5칸을 0으로 채운 배열 생성
# print(test_array4)
# print(type(test_array4))
#
# test_array5 = np.empty((4,5))
# print(test_array5)
# print(type(test_array5))
#
# test_array6 = np.eye(3)
# print(test_array6)
#
# test_array7 =np.full((2,3),5)
# # 2행 3열에 데이터는 전부 5 채우겠다.
# print(test_array7)
#
# test_random=np.random.random()
# print(test_random)
# testX=np.random.rand(100)
# testY=np.random.rand(100)
# print("rand X:",testX)
# print("rand Y:",testY)
#
# test_array2 = np.random.random(12).reshpae(3,4)
# # test_random2 데이터에 numpy random 난수 12개를 만들어서
# # 3행 4열 배열로 생성하겠다.
# print(test_random2)
#
# test_random_int=np.random.randint(0,10,12).reshape(3,4)
# # 정수형태의 난수 0~10 까지 중12개를 만들어 3행 4열에 배치한다.
# print(test_random_int)
# test_random_int=np.random.randint(0,10,(3,4))
# print(test_random_int)
# listx=[1,2,3,4]
# x = listx.length
#
# # 배열관련 속성 ndim : 배열의 차원, shape : 각 차원의 크기, dtype: 요소 데이터 타입
# # 배열관련 함수 reshape(), flatten(), ravel(), sort()
#
# arrayA=np.array([1,2,3])
# print(arrayA.ndim,"차원 배열이다.")
# print(arrayA.shape,"각 차원의 크기 shape")
# print(arrayA.dtype,"요소 타입")
#
# # 8비트가 1바이트
# # 8비트 ㅈ의 8제곱
# # int32 32비트 2의 32제곱
# # int64 64비트 2의 64제곱
#
# arrayB=np.array((range(12))) # 1차원 배열
# arrayB2=arrayB.reshape(2,6) # 2차원 배열
# # 12개 요소 배열 만든 후 reshape 통해 2행 6열로 재구성
# print(arrayB,"reshape X")
# print(arrayB2,"reshape O")
# print(arrayB2[1,1]) #행,열
# print(arrayB2[1]) #행 실제로는 2행
# print(arrayB2[1:,1:]) # 행 열 각각 적용  (1: 1인덱스 행부터 , 1: 1인덱스 열부터)
# # 다차원 배열을 1차원으로 만드는 방법
# arrayB3=arrayB2.flatten()
# print(arrayB3,"flatten 적용 다차원->1차원")
#
# arrayM = np.array([1,2,3]) # 1차원 1행 데이터 3개
# arrayN = np.array([4,5,6]) # 1차원 1행 데이터 3개
# print(np.add(arrayM,arrayN)) # 배열의 함
# print(np.subtract(arrayM,arrayN)) # 배열의 빼기 연산
# print(np.multiply(arrayM,arrayN)) # 배열의 곱 연산
# print(np.divide(arrayM,arrayN)) # 배열의 나누기 연산
# # 연산 방식은 동일한 구조, 동일한 위치의 요소끼리 연산이 된다.
# array_random_x = np.random.randint(0,10,(3,4)) # 0~10사이 3 X 4 행열
# array_random_y = np.random.randint(0,10,(3,4)) # 0~10사이 3 X 4 행열
# print(array_random_y < array_random_x) # if x>y: 이와 비슷한 느낌
# # 비교 방식은 동일한 구성, 동일한 지리의 요소끼리 비교를 해서 True/False
# arrayMax1 = [2,5,1,6,3,4]
# arrayMax2 = [1,2,3,4,5,6]
# array_maximum = np.maxium(arrayMax1,arrayMax2)
# print(array_maximum)
#
# print(np.power(arrayMax1,arrayMax2))
# # 배열의 연산 함수 : 같은 구성, 같은 구조, 같은 자리끼리 연산, 비교가 된다.


# # 시리즈에 대한 연산
# print(series1*2) # 전체 요소에 연산이 적용되었다.
# print("\n"*3)
#
# # 시리즈 내부인덱스 검사 => True/False 리턴 , 내가 원하는 인덱스가 있는지 확인할 때
# xxx=input("검색할 인덱스를 입력하세요.")
# print("첫" in series1)
# print(xxx in series1)
# print("\n" *3)
#
#
# # ######################## 데이터 프레임 만들기 1 배열로 생성
# df1 =pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]])
# print(df1)
# print(type(df1))
# print("\n"*3)
# # ################### value data는 저장된 데이터를 2차원 배열로 리턴
# print(df1.values)
# print(type(df1.values))
# print("\n"*3)

########################## 데이터 프레임 만들기 2
names = ['철수', '영희', '민수', '영자']
eng_point =[90,85,70,75]
math_point = [100,90,90,85]
df2 = pd.DataFrame([x for x in zip(names, eng_point,math_point)]
                   , columns = ['학생', '영어', '수학'])

print(df2)
print("\n"*3)

print(df2.columns)
print("\n"*3)
########################## 데이터 프레임 만들기 3 딕셔너리
data3={
'city': ['서울','부산','대전'],
'people' : ['100','80','50'],
'width' : ['10','5','3']
}
df3 = pd.DataFrame(data3)
print(df3)
print(type(df3))
print("\n"*3)


# ######################## 데이터 프레임3에 컬럼 경
df3 = pd.DataFrame(data3, columns=['city','city2','city3','city4'])
print(df3)
print("\n"*3)

# ##################### 데이터 프레임3에 컬럼 추가
df= pd.DataFrame(data3, columns=['city','people','width','height'])
print(df3)
print("\n"*3)
# ###################### 데이터 프레임3에 컬럼, 인덱스 추가
df3 = pd.DataFrame(data3, columns=['city','people','width','height','number'],
                   index=["one","two","three"])
# index=["one","two","three"]  => 프로퍼티명이 index 프로퍼티 밸류 ["one","two","three"]
print(df3)
print(df3.index) # 인덱스 프로퍼티 (인덱스 오브젝트)
print(df3.columns) # 컬럼 프로퍼티 (컬럼 오브젝트)
print("\n"*3)
#
# ###########################################

print(data.head()) #상위 5개 행 출력
# 매개변수 1 넣을 경우 1개의 상위 행 출력
# 처음나오는 0은 0번 인덱스 행
print(data.tail()) #하위 5개 행 출력
# 매개변수 1넣을 경우 1개의 상위 행 출력 # 처음
# 처음나오는 4는 4번 인덱스 행
print("\n"*3)
print(data.index) # rangeindex 표현 start stop step
print("\n"*3)
print(data.values) # 표의 값만 표현
print("\n"*3)

data.info() #데이터 칼럼 수 , 데이터타입 메모리 사용량 등 전반적 데이터에 대한 정보 표현
print("\n"*3)

print(data.describe()) # 데이터의 컬럼별 요약 통계량 표현
# # mean, max, median 함수로 통계란 계산 가능.
print("\n"*3)

print(data.mean()) #평균 값 출력
print("\n"*3)

print(data.max()) #최대 값 출력
print("\n"*3)

print(data.min()) #최소 값 출력
print("\n"*3)

print(data.median())  #중앙값(median)은 전체 데이터 중 가운데에 있는 수이다. 데이터의 수가 짝수인 경우는 가장 가운데에 있는 두 수의 평균이 중앙값이다. 2 3 10 12 6 8 직원이
print("\n"*3)

print(data["사고건수"].mean()) # 해당 컬럼의 mean 값만 출력
print("\n"*3)

print(data["사고건수"].count()) # 카운트
print("\n"*3)

print(data.sort_values(by='사망자수')) # 해당 value에대한 오름차순 정렬
print("\n"*3)

print(data.sort_values(by='사망자수', ascending=False)) # ascending=False=>내림차수
print("\n"*3)

print(data.sort_values(by='사망자수').tail(2))
print("\n"*3)

print(data["사망자수"])
print("\n"*3)

print(data[["사망자수", "경상자수"]]) #여러개의 컬럼 데이터를 추출하는 경우에는 대괄호를 두번([[ ]]) 사용합니다.
print("\n"*3)
#
sample1=data['사망자수']
print(sample1)
print(data["사망자수"].mean())
print(type(sample1)) #타입은 시리즈로 출력
print("\n"*3)
#

# ################# 데이터 슬라이싱

print(data[0:5])
print(data[1:3])
print(data[2:5])
print(data[4:6])
print("\n"*3)
#
#######################################
#데이터 프레임에 대한 인덱스 세팅과 재설정
data.set_index('발생년', inplace=True) #발생년을 인덱스로 설정하겠다.
print(data)
print("\n"*3)

print(data.loc['2016':'2017']) #행 데이터 조회
print("\n"*3)

print(data)

data1617 = data.loc['2016':'2017'] # 행 데이터 뽑아서 새 데이터로 만듦
print(data1617)
print("\n"*3)

x = data.loc['2016':'2018',['사망자수','사고건수']] #2016~2018년의 사망자수 사고건수 출력
print(x)
print(type(x)) #판다스 데이터 프레임
print("\n"*3)
#
data2= data.copy() #데이터 프레임 복사
data2['확인여부']=['1','1','1','1','1'] # 새로운 칼럼 '확인여부'에 데이터 1,1,1,1,1 지정
print(data2)

data2['중상및사망']= data2['사망자수']+data2['중상자수'] # 사망자수와 중상자수를 더해서 중상및 사망으로 새 컬럼 생성
data2['사고시사망활률'] = (data2['사망자수']/data2['사고건수'])*100 # 사망자수와 사고건수 연산하여 사망활률계산
print(data2)

data2.to_excel('data22.xlsx') # 엑셀로 저장
data2.to_html('data22.html') # html로 저장

data2.index=['2015y','2016y','2017y','2018y','2019y']
print(data2)
data2.to_html('data1.html')

data2.sort_values(by='사망자수', ascending=False)

#################################################### 시각화
data2['사망자수'].plot(kind='barh',grid=True, figsize=(10,5)) # 바형태 그리드 선이 있게 사이즈 10 5
plt.show() #plot으로 그린 시각화 디자인 표시
#
# ################### 리스트 직접 만들어서 시각화도 가능하다.
xx=[10,200,2,31,4,58,6,6,34,35,3,2,3,4,200]
plt.figure(figsize=(10, 10))
plt.plot(xx,color='red',linestyle='dashed',marker='o'
         ,markerfacecolor='red',markersize='10')
plt.show()
# 마커
# '_' 실선
# '.' 포인트마커
# 'o' d원형마커
#D
#d
#^ 세모 위
#> 세모 우
#< 세모 좌
#
# figure.figzie 그림의 크기 가로세로 인치단위
# lines.linewidth 선의두께
# plt.rcParams["figure.figzie"=(10,5)]

################## 산점도뿌리다.
yy=[1,2,3,4,5,6,6,6,7,8,9,10,11,12,13]
plt.scatter(xx,yy) #x축,y축 매개변수 # scatter 산점도 표현
plt.show()


################## x축, y축에 대한 설정을 따로 한 후 grid() 하여 show
plt.scatter(data["사망자수"], data["중상자수"], color='grey',alpha=.5)
plt.xlabel("xxxxxxx", fontsize=13)
plt.ylabel("yyyyyyy", fontsize=13)
plt.show()

time=[2010,2011,2012,2013,2014,2015,2016,2017,2018]
data3=[1,3,2,42,32,44,22,22,33]
plt.figure()
plt.stem(time,data3) # 2개의 데이터 적용
plt.show()

# # plot() 라인플롯
# # stem() 스템플롯
# # boxplot() 박스플롯
# # scatter() 스캐터
# # 서피스
# #
# #hist() 히스토그램 차트
#################색상 레드 라인스타일 --
plt.plot([1,2,3],[6,20,12], color='red', linestyle='--')
plt.show()