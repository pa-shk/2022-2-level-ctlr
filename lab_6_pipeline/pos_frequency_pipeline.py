"""
Implementation of POSFrequencyPipeline for score ten only.
"""
import re
from typing import Optional
from pathlib import Path
from core_utils.article.article import Article, ArtifactType
from lab_6_pipeline.pipeline import ConlluToken, CorpusManager, MorphologicalTokenDTO, ConlluSentence
from core_utils.article.ud import extract_sentences_from_raw_conllu
from core_utils.constants import ASSETS_PATH
from core_utils.visualizer import visualize


def from_conllu(path: Path, article: Optional[Article] = None) -> Article:
    """
    Populates the Article abstraction with all information from the conllu file
    """
    with open(path, encoding='utf-8') as f:
        content = f.read()
    extracted_senteces = extract_sentences_from_raw_conllu(content)
    for sentences in extracted_senteces:
        conllu_tokens = []
        for token in sentences['tokens']:
            params = token.split('\t')
            position = params[0]
            text = params[1]
            lemma = params[2]
            pos = params[3]
            conllu_token = ConlluToken(text)
            conllu_token.set_position(position)
            morph_params = MorphologicalTokenDTO(lemma, pos)
            conllu_token.set_morphological_parameters(morph_params)
            conllu_tokens.append(conllu_token)
        sentences['tokens'] = conllu_tokens
    conllu_senteces = [ConlluSentence(**senteces) for senteces in extracted_senteces]
    if article:
        article.set_conllu_sentences(conllu_senteces)
    else:
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
            path = article.get_file_path(ArtifactType.POS_CONLLU)
            article = from_conllu(path)
            frequencies = self._count_frequencies(article)
            article.set_pos_info(frequencies)
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
