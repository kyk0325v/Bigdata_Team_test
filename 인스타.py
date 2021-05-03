from selenium import webdriver #셀레니움
from selenium.webdriver.common.keys import Keys
import time#시간지연 구현을 위해 타임모듈 임포트
import urllib.request #url 라이브러리에 서브패키지 리퀘스트
import datetime
import time
import selenium
#구글 크롬드라이버 검색 chromedriver
#크롬 우상단 크롬정보 버전 85.0 xxx 버전 맞는 것 다운로드
driver=webdriver.Chrome('C:\chromedriver.exe')#드라이버 경로
driver.implicitly_wait(2)#대기시간

driver.get('https://www.instagram.com/')#인스타 홈페이지 주소
element=driver.find_element_by_name("username")#네임 클래스 id 등으로 검색 가능. q를 입력하는 이유는 html상 name이 q로 지정되어있다.
# 사이트를 오픈했고 검색창이 뭔지 찾아냈다.

element=driver.find_element_by_xpath("/html/body/div[1]/section/main/article/div[2]/div[1]/div/form/div/div[5]/button/span[2]") #페이스북 접근
element.click()
time.sleep(3)
element=driver.find_element_by_xpath("/html/body/div[1]/div[3]/div[1]/div/div/div[2]/div[1]/form/div/div[1]/input") #페이스북 로그인창 접근
element.send_keys("kyk0325v@nate.com")
element=driver.find_element_by_xpath("/html/body/div[1]/div[3]/div[1]/div/div/div[2]/div[1]/form/div/div[2]/input") #페이스북 비밀번호창 접근
element.send_keys("Kimglory0325@")
element=driver.find_element_by_xpath("/html/body/div[1]/div[3]/div[1]/div/div/div[2]/div[1]/form/div/div[3]/button") #페이스북 로그인 버튼
element.click()
time.sleep(8)
element=driver.find_element_by_xpath("/html/body/div[4]/div/div/div/div[3]/button[2]") # 인스타 검색창 버튼
element.send_keys(Keys.RETURN)
element=driver.find_element_by_xpath("/html/body/div[1]/section/nav/div[2]/div/div/div[2]/input") # 인스타 검색전 접근)
element.send_keys("#마케팅")
element=driver.find_element_by_xpath("/html/body/div[1]/section/nav/div[2]/div/div/div[2]/div[3]/div/div[2]/div") # 메뉴바 접근
element=driver.find_element_by_xpath("/html/body/div[1]/section/nav/div[2]/div/div/div[2]/div[3]/div/div[2]/div/div[1]/a/div/div[2]/div[1]/div") # 메뉴바 검색어 버튼 
element.click()
time.sleep(3)
element=driver.find_element_by_xpath("/html/body/div[1]/section/main/article/div[1]/div/div/div[1]/div[1]/a/div") # 첫번쨰 사진 클릭
element.click()
time.sleep(3)

while True:
    time.sleep(3)
    element = driver.find_element_by_css_selector("._65Bje") # 이미지 넘김버튼
    element.click(random())



# def doScrollDown(whileSeconds):
#     start = datetime.datetime.now()
#     print("s",start)
#     end = start + datetime.timedelta(seconds=whileSeconds)
#     print("e",end)
#     while True:
#         driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')#자바스크립트 실행 윈도우 스크롤함수
#         time.sleep(1)
#         if datetime.datetime.now() > end:
#             break
#
# doScrollDown(3)
# #
# images=driver.find_elements_by_css_selector("._9AhH0")#.rg_i.q4luwd 이미지 클래스명
# count=1


# for image in images:
#     image.click()#포문 요소 image 반복 이미지클릭
#     time.sleep(2)#2초지연
#     imgurl = driver.find_element_by_xpath("./html/body/div[3]/div[2]/div/div[1]/section/div/div[2]/div/div[1]/div[1]/div[1]/div/div/div[1]/div[1]/img").get_attribute("src")#큰이미지의 클래스네임 .n3~~ 속성 가져오기 src 주소
# #f12 개발자도구 full xpath 가 더 정확하다.
#     urllib.request.urlretrieve(imgurl, str(count)+".jpg")#파일 다운로드 코드
#     count+=1


