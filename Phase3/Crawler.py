from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class Crawler:
    def __init__(self, start_file, limit):
        self.start_file = start_file
        self.limit = limit
        chrome_options = Options()
        chrome_options.add_argument("--window-size=1920,1500")
        self.driver = webdriver.Chrome(options=chrome_options, executable_path="chromedriver.exe")

    def start_crawling(self, x):
        url = "https://academic.microsoft.com/paper/" + x
        self.driver.get(url)
        try:
            element = WebDriverWait(self.driver, 10).until(EC.text_to_be_present_in_element(
                (By.XPATH,
                 r'//*[@id="mainArea"]/router-view/router-view/ma-edp-serp/div/div[2]/div/compose/div/div[2]/ma-card[1]/div/compose/div/div[1]/a[1]')))
        finally:
            print("sss")
        year = self.driver.find_element_by_xpath('//*[@id="mainArea"]/router-view/div/div/div/div/a[1]/span[1]').text
        title = self.driver.find_element_by_xpath('//*[@id="mainArea"]/router-view/div/div/div/div/h1').text
        abstract = self.driver.find_element_by_xpath('//*[@id="mainArea"]/router-view/div/div/div/div/p').text
        authors = self.driver.find_element_by_class_name("authors").find_elements_by_tag_name("div")
        authors = [el.find_element_by_tag_name("a").text for el in authors]
        refrences = self.driver.find_element_by_xpath('//*[@id="mainArea"]/router-view/router-view/ma-edp-serp/div/div[2]/div/compose/div/div[2]') \
            .find_elements_by_tag_name("ma-card")
        refrences = [el.find_element_by_xpath("div/compose/div/div[1]/a[1]").get_attribute("href") for el in refrences]
        print(year)
        print(title)
        print(abstract)
        print(authors)
        print(refrences)

    def close_crawler(self):
        self.driver.close()


# elements = driver.find_elements_by_css_selector(".storylink")
# storyTitles = [el.text for el in elements]
# storyUrls = [el.get_attribute("href") for el in elements]
# elements = driver.find_elements_by_css_selector(".score")
# scores = [el.text for el in elements]
# elements = driver.find_elements_by_css_selector(".sitebit a")
# sites = [el.get_attribute("href") for el in elements]


c = Crawler("", 1)
c.start_crawling("3105081694")
c.close_crawler()
print("OK")
