from selenium import webdriver #셀레니움
from selenium.webdriver.common.keys import Keys
import time#시간지연 구현을 위해 타임모듈 임포트
import urllib.request #url 라이브러리에 서브패키지 리퀘스트
import datetime
import selenium
#구글 크롬드라이버 검색 chromedriver
#크롬 우상단 크롬정보 버전 85.0 xxx 버전 맞는 것 다운로드
driver=webdriver.Chrome('C:\chromedriver.exe')#드라이버 경로
driver.implicitly_wait(2)#대기시간

driver.get('https://www.google.co.kr/imghp?hl=ko&ogbl')#구글 이미지 주소 입력
element=driver.find_element_by_name("q")#네임 클래스 id 등으로 검색 가능. q를 입력하는 이유는 html상 name이 q로 지정되어있다.
# 사이트를 오픈했고 검색창이 뭔지 찾안냈다.


#html 설명
element.send_keys("") #키보드 입력값
element.send_keys(Keys.RETURN) #엔터키 = 리턴

def doScrollDown(whileSeconds):
    start = datetime.datetime.now()
    print("s",start)
    end = start + datetime.timedelta(seconds=whileSeconds)
    print("e",end)
    while True:
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')#자바스크립트 실행 윈도우 스크롤함수
        time.sleep(1)
        if datetime.datetime.now() > end:
            break

doScrollDown(3)

images=driver.find_elements_by_css_selector(".rg_i.Q4LuWd")#.rg_i.q4luwd 이미지 클래스명
count=1


for image in images:
    image.click()#포문 요소 image 반복 이미지클릭
    time.sleep(2)#2초지연
    imgurl = driver.find_element_by_css_selector(".n3VNCb").get_attribute("src")#큰이미지의 클래스네임 .n3~~ 속성 가져오기 src 주소
#f12 개발자도구 full xpath 가 더 정확하다.
    urllib.request.urlretrieve(imgurl, str(count)+".jpg")#파일 다운로드 코드
    count+=1
