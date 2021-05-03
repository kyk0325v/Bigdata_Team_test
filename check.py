from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import time


def beyondAlert( browser, delay ):
	try:
		# Alert 등 공지가 있는 경우 자동 확인
		WebDriverWait( browser, delay).until( EC.alert_is_present() )
		alert = browser.switch_to_alert()
		alert.accept()
		print("alert accepted")
	except TimeoutException:
		print("no alert")

	browser.switch_to_default_content()

def execute_myscript( browser ):
	javascript_scrtip = "javascript:setWorkTime('S');"
	browser.execute_script( javascript_scrtip )

def checkWork( browser, delay ):
	try:
		WebDriverWait( browser, delay ).until( EC.presence_of_element_located( (By.NAME, 'topFrame') ) )
		browser.switch_to.frame("topFrame")
		execute_myscript( browser )
	except TimeoutException:
		print("Loading took too much time!")

def main():

	mycompanyUrl = "http://kms.mycompany.com/index.do"
	loginIdElementName = 'LoginId'
	myId = 'JohnDoe'
	passwordElementName = 'LoginPwd'
	myPwd = 'mysecretpassword'

	#https://stackoverflow.com/a/26567563

	#browser = webdriver.Ie()
	browser = webdriver.Chrome()

	# LOGIN PAGE
	browser.get(mycompanyUrl)
	id = browser.find_element_by_name( loginIdElementName )
	id.send_keys( myId )

	pw = browser.find_element_by_name( passwordElementName )
	pw.send_keys( myPwd + Keys.RETURN )

	# MAIN PAGE
	delay = 3 # seconds

	beyondAlert( browser, delay )
	checkWork( browser, delay )


# https://stackoverflow.com/a/8054179
import logging
import sys

logger = logging.getLogger('mylogger')
# Configure logger to write to a file...

fileHandler = logging.FileHandler('./myLoggerTest.log')
logger.addHandler(fileHandler)

def my_handler(type, value, tb):
    logger.exception("Uncaught exception: {0}".format(str(value)))

# Install exception handler
sys.excepthook = my_handler

# Run your main script here:
if __name__ == '__main__':
    main()
