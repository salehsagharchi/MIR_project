import pickle
import os, json

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from Phase3.DataModels import Paper
from Phase3 import Constants


class Crawler:
    def __init__(self, start_file, limit, result_path, prior_file):
        self.start_file = start_file
        self.limit = limit
        self.errors = []
        if not os.path.isdir(result_path):
            os.makedirs(result_path)
        self.json_result_path = os.path.join(result_path, "result.json")
        self.crawling_state_path = os.path.join(result_path, "state")
        with open(self.start_file, 'r') as file:
            start_links = file.readlines()
        self.url_queue = [el.strip().split("/")[4] for el in start_links]
        self.processed_urls = []
        self.processed_papers = []
        self.cant_processed_urls = []
        self.prior_papers = dict()
        chrome_options = Options()
        chrome_options.add_argument("--window-size=500,500")
        self.driver = webdriver.Chrome(options=chrome_options, executable_path="chromedriver.exe")
        if prior_file:
            with open(prior_file, "r") as file:
                prior_json = "".join(file.readlines())
            prior_json_dict: dict = json.loads(prior_json)
            for paper in prior_json_dict['papers']:
                self.prior_papers[paper['id']] = Crawler.convert_dict_to_paper(paper)

    def start_crawling(self):
        success = "\033[32m" + "SUCCESSFUL" + "\033[0m"
        fail = "\033[91m" + "FAIL" + "\033[0m"
        i = 0
        while len(self.processed_urls) < self.limit:
            if len(self.url_queue) == 0:
                print("URL QUEUE IS EMPTY :/")
                break
            newurl = self.url_queue.pop(0)
            if newurl in self.processed_urls:
                continue
            i += 1
            print(f"Crawling paper #{i} with ID={newurl}", end="")
            newpaper, errors = self.crawl_a_page(newurl)
            if newpaper is not None:
                self.processed_urls.append(newurl)
                self.processed_papers.append(newpaper)
                for ref in newpaper.references:
                    if (ref not in self.processed_urls) and (ref not in self.url_queue):
                        self.url_queue.append(ref)
                print(f"\t{success}")
            else:
                if newurl not in self.cant_processed_urls:
                    self.cant_processed_urls.append(newurl)
                print(f"\t{fail}")
            if errors:
                print("\033[91m", errors, "\033[0m")
                self.errors += errors
            with open(self.crawling_state_path, "wb") as file:
                pickle.dump([self.url_queue, self.processed_urls, self.processed_papers, self.errors, self.cant_processed_urls, i], file)

        json_papers = [el.__dict__ for el in self.processed_papers]
        json_data = json.dumps({'papers': json_papers, 'procecced_papers': self.processed_urls,
                                'cant_processed_papers': self.cant_processed_urls, 'errors': self.errors}, indent=4)
        with open(self.json_result_path, "w") as file:
            file.write(json_data)

        print(f"{len(self.processed_urls)} Papers Saved Successfully in {self.json_result_path} !")

    def crawl_a_page(self, paper_id):
        errors = []
        try:
            assert paper_id
            if paper_id in self.prior_papers:
                return self.prior_papers[paper_id], errors
            url = "https://academic.microsoft.com/paper/" + paper_id
            self.driver.get(url)
            try:
                element = WebDriverWait(self.driver, 15) \
                    .until(EC.visibility_of_element_located((By.XPATH, r'//*[@id="mainArea"]/router-view/router-view/ma-edp-serp/' +
                                                             r'div/div[2]/div/compose/div/div[2]/ma-card[1]/div/compose/div/div[1]/a[1]')))
            except:
                raise Exception(f"Timeout Execption in Doc {paper_id}")

            year = self.driver.find_element_by_xpath('//*[@id="mainArea"]/router-view/div/div/div/div/a[1]/span[1]').text
            title = self.driver.find_element_by_xpath('//*[@id="mainArea"]/router-view/div/div/div/div/h1').text
            abstract = self.driver.find_element_by_xpath('//*[@id="mainArea"]/router-view/div/div/div/div/p').text
            authors_div = self.driver.find_element_by_class_name("authors").find_elements_by_tag_name("div")
            authors = []
            for el in authors_div:
                try:
                    authors.append(el.find_element_by_tag_name("a").text)
                except:
                    error = f"Author read error in {paper_id}"
                    errors.append(error)
            refrences = self.driver.find_element_by_xpath('//*[@id="mainArea"]/router-view/router-view/ma-edp-serp/div/div[2]/div/compose/div/div[2]') \
                .find_elements_by_tag_name("ma-card")
            refrenceLinks = []
            for el in refrences:
                try:
                    refrenceLinks.append(el.find_element_by_xpath("div/compose/div/div[1]/a[1]").get_attribute("href"))
                except:
                    error = f"Error reading refrence in {paper_id}"
                    errors.append(error)
            if len(refrenceLinks) != 10:
                error = f"# of refrence in {paper_id} = {len(refrenceLinks)}"
                errors.append(error)
            refrenceLinks = [el.split("/")[4] for el in refrenceLinks]
            assert year, "year is Empty"
            assert title, "title is Empty"
            assert authors, "authors is Empty"
            paper = Paper(paper_id, title, abstract, year, authors, refrenceLinks)
            return paper, errors
        except Exception as err:
            error = f"Exception in doc {paper_id} : {type(err)} {err}"
            errors.append(error)
            return None, errors

    def close_crawler(self):
        self.driver.close()

    @staticmethod
    def convert_dict_to_paper(dict):
        return Paper(dict['id'], dict['title'], dict['abstract'], dict['date'], dict['authors'], dict['references'])


c = Crawler(Constants.crawler_start_file, 5000, Constants.crawler_data_dir_root, "")
c.start_crawling()
c.close_crawler()
print("OK")
