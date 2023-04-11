"""
Crawler implementation
"""
import datetime
import json
import re
import shutil
from pathlib import Path
from typing import Pattern, Union

import requests
from bs4 import BeautifulSoup

from core_utils.article.article import Article
from core_utils.article.io import to_meta, to_raw
from core_utils.config_dto import ConfigDTO
from core_utils.constants import (ASSETS_PATH, CRAWLER_CONFIG_PATH,
                                  NUM_ARTICLES_UPPER_LIMIT,
                                  TIMEOUT_LOWER_LIMIT, TIMEOUT_UPPER_LIMIT)


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
        config_dto = self._extract_config_content()

        if not isinstance(config_dto.seed_urls, list):
            raise IncorrectSeedURLError

        for url in config_dto.seed_urls:
            if not isinstance(url, str) or not re.match(r'https?://.*/', url):
                raise IncorrectSeedURLError

        if (not isinstance(config_dto.total_articles, int)
                or isinstance(config_dto.total_articles, bool)
                or config_dto.total_articles < 1):
            raise IncorrectNumberOfArticlesError

        if config_dto.total_articles > NUM_ARTICLES_UPPER_LIMIT:
            raise NumberOfArticlesOutOfRangeError

        if not isinstance(config_dto.headers, dict):
            raise IncorrectHeadersError

        if not isinstance(config_dto.encoding, str):
            raise IncorrectEncodingError

        if (not isinstance(config_dto.timeout, int)
                or config_dto.timeout < TIMEOUT_LOWER_LIMIT
                or config_dto.timeout > TIMEOUT_UPPER_LIMIT):
            raise IncorrectTimeoutError

        if (not isinstance(config_dto.should_verify_certificate, bool)
                or not isinstance(config_dto.headless_mode, bool)):
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
    response = requests.get(
        url,
        headers=config.get_headers(),
        timeout=config.get_timeout(),
        verify=config.get_verify_certificate()
    )
    response.encoding = config.get_encoding()
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
        self._seed_urls = config.get_seed_urls()
        self._config = config
        self.urls = []

    def _extract_url(self, article_bs: BeautifulSoup) -> str:
        """
        Finds and retrieves URL from HTML
        """
        url = article_bs.get('href')
        if isinstance(url, str):
            return url
        return ''

    def find_articles(self) -> None:
        """
        Finds articles
        """
        for seed_url in self._seed_urls:
            res = make_request(seed_url, self._config)
            soup = BeautifulSoup(res.content, "lxml")
            for paragraph in soup.find_all('a', class_="article-list__title"):
                if len(self.urls) >= self._config.get_num_articles():
                    return
                url = self._extract_url(paragraph)
                if not url or url in self.urls:
                    continue
                self.urls.append(url)

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
        self.article.text = ' '.join(i.text.strip() for i in text_paragraphs)

    def _fill_article_with_meta_information(self, article_soup: BeautifulSoup) -> None:
        """
        Finds meta information of article
        """
        title = article_soup.find('h1', class_="article__title")
        if title:
            self.article.title = title.text
        date = article_soup.find(class_="article__meta-date")
        if date:
            try:
                self.article.date = self.unify_date_format(date.text)
            except ValueError:
                pass
        topics = [topic.text for topic in article_soup.find_all('a', class_="article-list__tag")]
        if topics:
            self.article.topics = topics
        self.article.author = ["NOT FOUND"]

    def unify_date_format(self, date_str: str) -> datetime.datetime:
        """
        Unifies date format
        """
        if not re.search(r'\d{4}', date_str):
            curr_year = ' ' + str(datetime.date.today().year)
            date_str = re.sub(r'(?<=[А-Яа-я])(?=,\s\d{2})', curr_year, date_str)

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
        self._load_crawler_data()

    def _load_crawler_data(self) -> None:
        current_path = Path(__file__)
        crawler_data_path = current_path.parent / 'crawler_data.json'
        if crawler_data_path.exists():
            with open('crawler_data.json', 'r', encoding='utf-8') as f:
                crawler_data = json.load(f)
            self.urls = crawler_data['urls']
            self.start_url = crawler_data['start_url']

    def _save_crawler_data(self) -> None:
        crawler_data = {
            'start_url': self.start_url,
            'urls': self.urls
        }
        with open('crawler_data.json', 'w', encoding='utf-8') as f:
            json.dump(crawler_data, f, ensure_ascii=True, indent=4, separators=(', ', ': '))

    def find_articles(self) -> None:
        """
        Finds articles
        """
        res = make_request(self.start_url, self._config)
        article_bs = BeautifulSoup(res.content, "html.parser")
        for soup in (
                * article_bs.find_all('a', class_="article-list__title"),
                * article_bs.find_all('a', class_="article__embedded"),
                * article_bs.find_all('a', class_="card__title")
        ):
            if len(self.urls) >= self._config.get_num_articles():
                return
            try:
                url = self._extract_url(soup)
            except KeyError:
                continue
            if url in self.urls:
                continue
            self.urls.append(url)
            self.start_url = url
            self._save_crawler_data()
            self.find_articles()


def main() -> None:
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


def main_recursive() -> None:
    """
    Driver code for recursive crawling
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
    main()
