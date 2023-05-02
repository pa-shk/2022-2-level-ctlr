"""
Implementation of POSFrequencyPipeline for score ten only.
"""
import re
from pathlib import Path
from typing import Optional

from core_utils.article.article import Article, ArtifactType
from core_utils.article.io import from_meta, to_meta
from core_utils.article.ud import extract_sentences_from_raw_conllu
from core_utils.constants import ASSETS_PATH
from core_utils.visualizer import visualize
from lab_6_pipeline.pipeline import (ConlluSentence, ConlluToken,
                                     CorpusManager, MorphologicalTokenDTO)


class EmptyFileError(Exception):
    pass


def from_conllu(path: Path, article: Optional[Article] = None) -> Article:
    """
    Populates the Article abstraction with all information from the conllu file
    """
    with open(path, encoding='utf-8') as f:
        content = f.read()
    extracted_sentences = extract_sentences_from_raw_conllu(content)
    for sentence in extracted_sentences:
        sentence['tokens'] = [_parse_conllu_token(token) for token in  sentence['tokens']]
    conllu_senteces = [ConlluSentence(**sentence) for sentence in extracted_sentences]
    if not article:
        article_id = int(re.match(r'\d+', path.stem).group())
        article = Article(None, article_id)
    article.set_conllu_sentences(conllu_senteces)
    return article


def _parse_conllu_token(token_line: str) -> ConlluToken:
    """
    Parses the raw text in the CONLLU format into the CONLL-U token abstraction

    Example:
    '2	произошло	происходить	VERB	_	Gender=Neut|Number=Sing|Tense=Past	0	root	_	_'
    """
    params = token_line.split('\t')
    position = int(params[0])
    text = params[1]
    lemma = params[2]
    pos = params[3]
    token = ConlluToken(text)
    token.set_position(position)
    morph_params = MorphologicalTokenDTO(lemma, pos)
    token.set_morphological_parameters(morph_params)
    return token


# pylint: disable=too-few-public-methods
class POSFrequencyPipeline:
    """
    Counts frequencies of each POS in articles,
    updates meta information and produces graphic report
    """

    def __init__(self, corpus_manager: CorpusManager):
        """
        Initializes PosFrequencyPipeline
        """
        self._corpus = corpus_manager

    def run(self) -> None:
        """
        Visualizes the frequencies of each part of speech
        """
        for article in self._corpus.get_articles().values():
            conllu_path = article.get_file_path(ArtifactType.MORPHOLOGICAL_CONLLU)
            if not conllu_path.stat().st_size:
                raise EmptyFileError
            article = from_conllu(conllu_path)
            meta_file_path = article.get_meta_file_path()
            article = from_meta(meta_file_path, article)
            frequencies = self._count_frequencies(article)
            article.set_pos_info(frequencies)
            to_meta(article)
            visualize(article=article,
                      path_to_save=ASSETS_PATH / f'{article.article_id}_image.png')


    def _count_frequencies(self, article: Article) -> dict[str, int]:
        """
        Counts POS frequency in Article
        """
        frequencies = {}
        for sentences in article.get_conllu_sentences():
            for token in sentences.get_tokens():
                pos = token.get_morphological_parameters().pos
                frequencies[pos] = frequencies.get(pos, 0) + 1
        return frequencies


def main() -> None:
    """
    Entrypoint for the module
    """
    corpus_manager = CorpusManager(ASSETS_PATH)
    pipeline = POSFrequencyPipeline(corpus_manager=corpus_manager)
    pipeline.run()


if __name__ == "__main__":
    main()
