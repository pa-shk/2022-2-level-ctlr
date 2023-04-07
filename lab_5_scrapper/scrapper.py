"""
Crawler implementation
"""
import datetime
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Pattern, Union
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from core_utils.article.article import Article
from core_utils.article.io import to_raw, to_meta
from core_utils.config_dto import ConfigDTO
from core_utils.constants import (
    ASSETS_PATH,
    CRAWLER_CONFIG_PATH,
    NUM_ARTICLES_UPPER_LIMIT,
    TIMEOUT_LOWER_LIMIT,
    TIMEOUT_UPPER_LIMIT
)


class IncorrectSeedURLError(Exception):
    """
    Inappropriate value for seed url
    """


class NumberOfArticlesOutOfRangeError(Exception):
    """
    Number of articles either to large or small
    """


class IncorrectNumberOfArticlesError(Exception):
    """
    Inappropriate value for number of articles
    """


class IncorrectHeadersError(Exception):
    """
    Inappropriate value for headers
    """


class IncorrectEncodingError(Exception):
    """
    Inappropriate value for encoding
    """


class IncorrectTimeoutError(Exception):
    """
    Inappropriate value for timeout
    """


class IncorrectVerifyError(Exception):
    """
     Inappropriate value for certificate
     """


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
        config_dto = self._extract_config_content()
        self._seed_urls = config_dto.seed_urls
        self._num_articles = config_dto.total_articles
        self._headers = config_dto.headers
        self._encoding = config_dto.encoding
        self._timeout = config_dto.timeout
        self._should_verify_certificate = config_dto.should_verify_certificate
        self._headless_mode = config_dto.headless_mode


    def _extract_config_content(self) -> ConfigDTO:
        """
        Returns config values
        """
        with open(self.path_to_config, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return ConfigDTO(**config_dict)

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

        for url in seed_urls:
            if not isinstance(url, str)or not re.match(r'https?://.*/', url):
                raise IncorrectSeedURLError

        if (not isinstance(total_articles_to_find_and_parse, int)
                or isinstance(total_articles_to_find_and_parse, bool)
                or total_articles_to_find_and_parse < 1):
            raise IncorrectNumberOfArticlesError

        if total_articles_to_find_and_parse > NUM_ARTICLES_UPPER_LIMIT:
            raise NumberOfArticlesOutOfRangeError

        if not isinstance(headers, dict):
            raise IncorrectHeadersError

        if not isinstance(encoding, str):
            raise IncorrectEncodingError

        if (not isinstance(timeout, int)
                or timeout < TIMEOUT_LOWER_LIMIT
                or timeout > TIMEOUT_UPPER_LIMIT):
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
    determined_pause = 0.5
    divider = 2
    time.sleep(determined_pause + random.random() / divider)
    headers = config.get_headers()
    timeout = config.get_timeout()
    return requests.get(url, headers=headers, timeout=timeout)


class Crawler:
    """
    Crawler implementation
    """

    url_pattern: Union[Pattern, str]

    def __init__(self, config: Config) -> None:
        """
        Initializes an instance of the Crawler class
        """
        self._seed_urls = config.get_seed_urls()
        self._config = config
        self.urls = []

    def _extract_url(self, article_bs: BeautifulSoup) -> str:
        """
        Finds and retrieves URL from HTML
        """
        all_links = []
        all_links.extend(article_bs.find_all('a', class_="article-list__title"))
        all_links.extend(article_bs.find_all('a', class_="article__embedded"))
        all_links.extend(article_bs.find_all('a', class_="card__title"))
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
            while len(self.urls) < self._config.get_num_articles():
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
        article_content = article_soup.find("div", class_="article__content")
        text_paragraphs = article_content.find_all("p")
        text = ''.join(i.text for i in text_paragraphs)
        self.article.text = re.sub(r'\n+|\s{2,}', ' ', text)

    def _fill_article_with_meta_information(self, article_soup: BeautifulSoup) -> None:
        """
        Finds meta information of article
        """
        title = article_soup.find('h1', class_="article__title")
        if title:
            self.article.title = title.text
        date = article_soup.find(class_="article__meta-date")
        if date:
            date_text = date.text
            if not re.search(r'\d{4}', date_text):
                curr_year = ' ' + str(datetime.date.today().year)
                date_text = re.sub(r'(?<=[А-Яа-я])(?=,\s\d{2})', curr_year, date_text)
            try:
                self.article.date = self.unify_date_format(date_text)
            except ValueError:
                pass
        topics = [topic.text for topic in article_soup.find_all('a', class_="article-list__tag")]
        self.article.author = ["NOT FOUND"]
        if topics:
            self.article.topics = topics

    def unify_date_format(self, date_str: str) -> datetime.datetime:
        """
        Unifies date format
        """
        ru_eng_months = {
            "янв": "jan",
            "фев": "feb",
            "мар": "mar",
            "апр": "apr",
            "мая": "may",
            "июн": "jun",
            "июл": "jul",
            "авг": "aug",
            "сен": "sep",
            "окт": "oct",
            "ноя": "nov",
            "дек": "dec"
        }
        ru_month = re.search(r"[А-Яа-я]{3}", date_str).group()
        date_str = date_str.replace(ru_month, ru_eng_months[ru_month])
        return datetime.datetime.strptime(date_str, '%d %b  %Y, %H:%M')

    def parse(self) -> Union[Article, bool, list]:
        """
        Parses each article
        """
        page = make_request(self.full_url, self.config)
        article_bs = BeautifulSoup(page.content, "lxml")
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


class CrawlerRecursive(Crawler):
    """
    Recursive crawler implementation
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.start_url = config.get_seed_urls()[0]

    def find_articles(self) -> None:
        """
        Finds articles
        """
        if len(self.urls) >= self._config.get_num_articles():
            return
        res = make_request(self.start_url, self._config)
        soup = BeautifulSoup(res.content, "html.parser")
        page_urls = self._extract_url(soup)
        new_urls = []
        while len(self.urls) < self._config.get_num_articles():
            try:
                new_link = next(page_urls)
                new_urls.append(new_link)
                self.urls.append(new_link)
            except StopIteration:
                break
        for url in new_urls:
            self.start_url = url
            self.find_articles()


def main_1() -> None:
    """
    Entrypoint for scrapper module
    """
    configuration = Config(path_to_config=CRAWLER_CONFIG_PATH)
    prepare_environment(ASSETS_PATH)
    crawler = Crawler(config=configuration)
    crawler.find_articles()
    for i, url in enumerate(crawler.urls, start=1):
        parser = HTMLParser(full_url=url, article_id=i, config=configuration)
        article = parser.parse()
        if isinstance(article, Article):
            to_raw(article)
            to_meta(article)


def main_2() -> None:
    """
    Entrypoint for scrapper module
    """
    configuration = Config(path_to_config=CRAWLER_CONFIG_PATH)
    prepare_environment(ASSETS_PATH)
    crawler_recursive = CrawlerRecursive(config=configuration)
    crawler_recursive.find_articles()
    for i, url in enumerate(crawler_recursive.urls, start=1):
        parser = HTMLParser(full_url=url, article_id=i, config=configuration)
        article = parser.parse()
        if isinstance(article, Article):
            to_raw(article)
            to_meta(article)


if __name__ == "__main__":
    main_1()
