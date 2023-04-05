"""
Crawler implementation
"""
import time
from typing import Pattern, Union
from pathlib import Path
from core_utils.constants import CRAWLER_CONFIG_PATH, ASSETS_PATH
from core_utils.config_dto import ConfigDTO
import requests
from bs4 import BeautifulSoup
import datetime
from core_utils.article.article import Article
from core_utils.article.io import to_raw, to_meta
import json
import re
import shutil
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from urllib.parse import urlparse, urlunparse
import datetime
import locale
import re
# for German locale

#   datetime.datetime.strptime() method.


class IncorrectSeedURLError(Exception):
    pass


class NumberOfArticlesOutOfRangeError(Exception):
    pass


class IncorrectNumberOfArticlesError(Exception):
    pass


class IncorrectHeadersError(Exception):
    pass


class IncorrectEncodingError(Exception):
    pass


class IncorrectTimeoutError(Exception):
    pass


class IncorrectVerifyError(Exception):
    pass


class Config:
    """
    Unpacks and validates configurations
    """

    def __init__(self, path_to_config: Path) -> None:
        """
        Initializes an instance of the Config class
        """
        self.path_to_config = path_to_config
        self._validate_config_content()
        self._extract_config_content()


    def _extract_config_content(self) -> ConfigDTO:
        """
        Returns config values
        """
        with open(self.path_to_config, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        self._seed_urls =  config_dict['seed_urls']
        self._num_articles = config_dict['total_articles_to_find_and_parse']
        self._headers = config_dict['headers']
        self._encoding = config_dict['encoding']
        self._timeout = config_dict['timeout']
        self._should_verify_certificate = config_dict['should_verify_certificate']
        self._headless_mode = config_dict['headless_mode']

        config_instance = ConfigDTO(seed_urls=self._seed_urls,
                                    headers=self._headers,
                                    total_articles_to_find_and_parse=self._num_articles,
                                    encoding=self._encoding,
                                    timeout=self._timeout,
                                    should_verify_certificate=self._should_verify_certificate,
                                    headless_mode=self._headless_mode)

        return config_instance

    def _validate_config_content(self) -> None:
        """
        Ensure configuration parameters
        are not corrupt
        """
        with open(self.path_to_config, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        seed_urls = config_dict['seed_urls']
        headers = config_dict['headers']
        total_articles_to_find_and_parse = config_dict['total_articles_to_find_and_parse']
        encoding = config_dict['encoding']
        timeout = config_dict['timeout']
        verify_certificate = config_dict['should_verify_certificate']
        headless_mode = config_dict['headless_mode']

        if not isinstance(seed_urls, list):
            raise IncorrectSeedURLError
        if not all([re.match(r'https?://.*/', url) for url in seed_urls]):
            raise IncorrectSeedURLError
        if (not isinstance(total_articles_to_find_and_parse, int)
                or isinstance(total_articles_to_find_and_parse, bool)
                or total_articles_to_find_and_parse < 1):
            raise IncorrectNumberOfArticlesError
        if  total_articles_to_find_and_parse > 150:
            raise NumberOfArticlesOutOfRangeError
        if not isinstance(headers, dict):
            raise IncorrectHeadersError
        if not isinstance(encoding, str):
            raise IncorrectEncodingError
        if not isinstance(timeout, int) or  timeout < 0 or timeout > 60:
            raise IncorrectTimeoutError
        if not isinstance(verify_certificate, bool) or not isinstance(headless_mode, bool):
            raise IncorrectVerifyError

    def get_seed_urls(self) -> list[str]:
        """
        Retrieve seed urls
        """
        return self._seed_urls

    def get_num_articles(self) -> int:
        """
        Retrieve total number of articles to scrape
        """
        return self._num_articles

    def get_headers(self) -> dict[str, str]:
        """
        Retrieve headers to use during requesting
        """
        return self._headers

    def get_encoding(self) -> str:
        """
        Retrieve encoding to use during parsing
        """
        return self._encoding

    def get_timeout(self) -> int:
        """
        Retrieve number of seconds to wait for response
        """
        return self._timeout

    def get_verify_certificate(self) -> bool:
        """
        Retrieve whether to verify certificate
        """
        return self._should_verify_certificate

    def get_headless_mode(self) -> bool:
        """
        Retrieve whether to use headless mode
        """
        return self._headless_mode


def make_request(url: str, config: Config) -> requests.models.Response:
    """
    Delivers a response from a request
    with given configuration
    """
    time.sleep(config._timeout)
    response = requests.get(url,  headers=config._headers)
    return response


class Crawler:
    """
    Crawler implementation
    """

    url_pattern: Union[Pattern, str]

    def __init__(self, config: Config) -> None:
        """
        Initializes an instance of the Crawler class
        """
        self._seed_urls = config._seed_urls
        self._config = config
        self.urls = []

    def _extract_url(self, article_bs: BeautifulSoup) -> str:
        """
        Finds and retrieves URL from HTML
        """
        all_links = article_bs.find_all('a', class_="article-list__title")
        for link in all_links:
            try:
                address = link['href']
            except KeyError:
                continue
            yield address


    def find_articles(self) -> None:
        """
        Finds articles
        """
        for url in self._seed_urls:
            res = make_request(url, self._config)
            soup = BeautifulSoup(res.content, "html.parser")
            new_urls = self._extract_url(soup)
            while len(self.urls) < self._config._num_articles:
                try:
                    self.urls.append(next(new_urls))
                except StopIteration:
                    break


    def get_search_urls(self) -> list:
        """
        Returns seed_urls param
        """
        return self._seed_urls


class HTMLParser:
    """
    ArticleParser implementation
    """

    def __init__(self, full_url: str, article_id: int, config: Config) -> None:
        """
        Initializes an instance of the HTMLParser class
        """
        self.full_url = full_url
        self.article_id = article_id
        self.config = config
        self.article = Article(self.full_url, self.article_id)

    def _fill_article_with_text(self, article_soup: BeautifulSoup) -> None:
        """
        Finds text of article
        """
        text_paragraphs = article_soup.find_all("div", class_="article__content")
        self.article.text = ''.join(i.text for i in text_paragraphs)

    def _fill_article_with_meta_information(self, article_soup: BeautifulSoup) -> None:
        """
        Finds meta information of article
        """
        title = article_soup.find('h1', class_="article__title")
        if title:
            self.article.title = title.text
        day_month = article_soup.find(class_="article__meta-date")
        year = article_soup.find(class_="footer__copyright")
        invalid_year = '1000'
        year = re.search(r'\d{4}', year.text) if year else None
        year = year.group() if year else None
        if not year:
            year = invalid_year
        if day_month:
            date = ' '.join((day_month.text, year))
            self.article.date = self.unify_date_format(date)
        topics = [topic.text for topic in article_soup.find_all('a', class_="article-list__tag")
              if topic]
        self.article.author = ["NOT FOUND"]
        if topics:
            self.article.topics = topics

    def unify_date_format(self, date_str: str) -> datetime.datetime:
        """
        Unifies date format
        """
        locale.setlocale(locale.LC_TIME, "ru_RU")
        return datetime.datetime.strptime(date_str, '%d %b, %H:%M %Y')

    def parse(self) -> Union[Article, bool, list]:
        """
        Parses each article
        """
        page = make_request(self.full_url, self.config)
        article_bs = BeautifulSoup(page.content, "html.parser")
        self._fill_article_with_text(article_bs)
        self._fill_article_with_meta_information(article_bs)
        return self.article


def prepare_environment(base_path: Union[Path, str]) -> None:
    """
    Creates ASSETS_PATH folder if no created and removes existing folder
    """
    if base_path.exists():
        shutil.rmtree(base_path)
    base_path.mkdir(parents=True)


def main() -> None:
    """
    Entrypoint for scrapper module%b
    """
    configuration = Config(path_to_config=CRAWLER_CONFIG_PATH)
    configuration._validate_config_content()
    configuration._extract_config_content()
    prepare_environment(ASSETS_PATH)
    crawler = Crawler(config=configuration)
    crawler.find_articles()
    for i, url in enumerate(crawler.urls, start=1):
        parser = HTMLParser(full_url=url, article_id=i, config=configuration)
        article = parser.parse()
        to_raw(article)
        to_meta(article)


if __name__ == "__main__":
    main()
