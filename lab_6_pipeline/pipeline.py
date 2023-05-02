"""
Pipeline for CONLL-U formatting
"""
from pathlib import Path
import string
from typing import List
import re

import pymorphy2
from pymystem3 import Mystem

from core_utils.article.article import SentenceProtocol, Article, split_by_sentence
from core_utils.article.io import to_cleaned, from_raw, to_conllu
from core_utils.article.ud import OpencorporaTagProtocol, TagConverter
from core_utils.constants import ASSETS_PATH



# pylint: disable=too-few-public-methods
class InconsistentDatasetError(Exception):
    pass


class EmptyDirectoryError(Exception):
    pass

class CorpusManager:
    """
    Works with articles and stores them
    """

    def __init__(self, path_to_raw_txt_data: Path):
        """
        Initializes CorpusManager
        """
        self.path_to_raw_txt_data = path_to_raw_txt_data
        self._validate_dataset()
        self._storage = {}
        self._scan_dataset()

    def _validate_dataset(self) -> None:
        """
        Validates folder with assets
        """
        if not self.path_to_raw_txt_data.exists():
            raise FileNotFoundError
        if not self.path_to_raw_txt_data.is_dir():
            raise NotADirectoryError
        if not any(self.path_to_raw_txt_data.iterdir()):
            raise  EmptyDirectoryError
        meta_files = [i for i in self.path_to_raw_txt_data.glob('**/*.json') if  re.match(r'\d+_meta', i.stem)]
        text_files = [i for i in self.path_to_raw_txt_data.glob('**/*.txt') if re.match(r'\d+_raw', i.stem)]
        if len(meta_files) != len(text_files):
            raise InconsistentDatasetError
        for files in meta_files, text_files:
            if sorted(int(re.search(r'\d+', i.stem)[0]) for i in files) != list(range(1, len(files) + 1)):
                raise InconsistentDatasetError
            if not all(i.stat().st_size for i in files):
                raise InconsistentDatasetError

    def _scan_dataset(self) -> None:
        """
        Register each dataset entry
        """
        for path in self.path_to_raw_txt_data.glob('**/*.txt'):
            if not (relevant:= re.search(r'(\d+)_raw', path.stem)):
                continue
            self._storage[int(relevant[1])] = from_raw(path)

    def get_articles(self) -> dict:
        """
        Returns storage params
        """
        return self._storage

class MorphologicalTokenDTO:
    """
    Stores morphological parameters for each token
    """

    def __init__(self, lemma: str = "", pos: str = "", tags: str = ""):
        """
        Initializes MorphologicalTokenDTO
        """
        self.lemma = lemma
        self.pos = pos
        self.tags = tags

class ConlluToken:
    """
    Representation of the CONLL-U Token
    """

    def __init__(self, text: str):
        """
        Initializes ConlluToken
        """
        self._text = text
        self._morphological_parameters = MorphologicalTokenDTO()
        self._position = 0

    def set_morphological_parameters(self, parameters: MorphologicalTokenDTO) -> None:
        """
        Stores the morphological parameters
        """
        self._morphological_parameters = parameters

    def set_position(self, postion: int) -> None:
        self._position = postion

    def get_morphological_parameters(self) -> MorphologicalTokenDTO:
        """
        Returns morphological parameters from ConlluToken
        """
        return self._morphological_parameters

    def get_conllu_text(self, include_morphological_tags: bool) -> str:
        """
        String representation of the token for conllu files
        """
        position = str(self._position)
        text = self._text
        lemma = self._morphological_parameters.lemma
        pos = self._morphological_parameters.pos
        xpos = '_'
        feats = '_'
        if include_morphological_tags:
            if tags := self._morphological_parameters.tags:
                feats = tags
        head = '0'
        deprel = 'root'
        deps = '_'
        misc = '_'

        return '\t'.join([position, text, lemma, pos, xpos, feats, head, deprel, deps, misc])

    def get_cleaned(self) -> str:
        """
        Returns lowercase original form of a token
        """
        return re.sub(r'[^\w\s]+', '', self._text).lower()

class ConlluSentence(SentenceProtocol):
    """
    Representation of a sentence in the CONLL-U format
    """

    def __init__(self, position: int, text: str, tokens: list[ConlluToken]):
        """
        Initializes ConlluSentence
        """
        self._position = position
        self._text = text
        self._tokens = tokens

    def get_conllu_text(self, include_morphological_tags: bool) -> str:
        """
        Creates string representation of the sentence
        """
        return self._format_tokens(include_morphological_tags)

    def get_cleaned_sentence(self) -> str:
        """
        Returns the lowercase representation of the sentence
        """
        return ' '.join(filter(bool, (i.get_cleaned() for i in self._tokens)))

    def get_tokens(self) -> list[ConlluToken]:
        """
        Returns sentences from ConlluSentence
        """
        return self._tokens

    def _format_tokens(self, include_morphological_tags: bool) -> str:
        return (f'# sent_id = {self._position}\n# text = {self._text}\n'
                +  '\n'.join(i.get_conllu_text(include_morphological_tags) for i in self._tokens)
                + '\n')

class MystemTagConverter(TagConverter):
    """
    Mystem Tag Converter
    """

    def convert_morphological_tags(self, tags: str) -> str:  # type: ignore
        """
        Converts the Mystem tags into the UD format
        """
        tags = re.sub(r'\((.+?)\|.+\)', r'\1', tags)
        extracted_tags = re.findall(r'[а-я]+', tags)
        ud_tags = {}
        for tag in extracted_tags:
            for category in (self.case, self.number, self.gender, self.animacy, self.tense):
                if tag in self._tag_mapping[category]:
                    ud_tags[category] = self._tag_mapping[category][tag]
        return '|'.join(f'{k}={v}' for k, v in sorted(ud_tags.items()))


    def convert_pos(self, tags: str) -> str:  # type: ignore
        """
        Extracts and converts the POS from the Mystem tags into the UD format
        """
        pos = re.match(r'\w+', tags)[0]
        return self._tag_mapping[self.pos][pos]


class OpenCorporaTagConverter(TagConverter):
    """
    OpenCorpora Tag Converter
    """

    def convert_pos(self, tags: OpencorporaTagProtocol) -> str:  # type: ignore
        """
        Extracts and converts POS from the OpenCorpora tags into the UD format
        """
        return self._tag_mapping[self.pos][tags.POS]

    def convert_morphological_tags(self, tags: OpencorporaTagProtocol) -> str:  # type: ignore
        """
        Converts the OpenCorpora tags into the UD format
        """
        ud_tags = {}
        for category in ( 'animacy', 'case', 'gender', 'number'):
            if not (open_corpora_tag := eval(f'tags.{category}')):
                continue
            ud_tags[eval(f'self.{category}')] = eval(f"self._tag_mapping[self.{category}]['{open_corpora_tag}']")
        return '|'.join(f'{k}={v}' for k, v in ud_tags.items())


class MorphologicalAnalysisPipeline:
    """
    Preprocesses and morphologically annotates sentences into the CONLL-U format
    """

    def __init__(self, corpus_manager: CorpusManager):
        """
        Initializes MorphologicalAnalysisPipeline
        """
        self._corpus = corpus_manager
        self._stemmer = Mystem()
        mapping_path = Path(__file__).parent / 'data' / 'mystem_tags_mapping.json'
        self._converter = MystemTagConverter(mapping_path)

    def _process(self, text: str) -> List[ConlluSentence]:
        """
        Returns the text representation as the list of ConlluSentence
        """
        sentences = []
        result = self._stemmer.analyze(re.sub(r'\W+', ' ', text))
        token_count = 0
        for sentence_position, sentence in enumerate(split_by_sentence(text)):
            conllu_tokens = []
            for token_position, token in enumerate(re.findall(r'\w+', sentence), start=1):
                conllu_token = ConlluToken(token)
                if not result[token_count]['text'].isalnum():
                    token_count += 1
                if 'analysis' in result[token_count] and result[token_count]['analysis']:
                    lex = result[token_count]['analysis'][0]['lex']
                    pos = self._converter.convert_pos(result[token_count]['analysis'][0]['gr'])
                    tags = self._converter.convert_morphological_tags(result[token_count]['analysis'][0]['gr'])
                elif result[token_count]['text'].isdigit():
                    lex = result[token_count]['text']
                    pos = 'NUM'
                    tags = ''
                else:
                    lex = result[token_count]['text']
                    pos = 'X'
                    tags = ''
                morph_params = MorphologicalTokenDTO(lex, pos, tags)
                conllu_token.set_position(token_position)
                conllu_token.set_morphological_parameters(morph_params)
                conllu_tokens.append(conllu_token)
                token_count += 1
            end_token = ConlluToken('.')
            end_token.set_position(token_position + 1)
            morph_params = MorphologicalTokenDTO('.',  'PUNCT')
            end_token.set_morphological_parameters(morph_params)
            conllu_tokens.append(end_token)
            sentence = ConlluSentence(sentence_position, sentence, conllu_tokens)
            sentences.append(sentence)
        return sentences

    def run(self) -> None:
        """
        Performs basic preprocessing and writes processed text to files
        """
        for article in self._corpus.get_articles().values():
            article.set_conllu_sentences(self._process(article.text))
            to_cleaned(article)
            to_conllu(article,
                      include_morphological_tags=False,
                      include_pymorphy_tags=False)
            to_conllu(article,
                      include_morphological_tags=True,
                      include_pymorphy_tags=False)

class AdvancedMorphologicalAnalysisPipeline(MorphologicalAnalysisPipeline):
    """
    Preprocesses and morphologically annotates sentences into the CONLL-U format
    """

    def __init__(self, corpus_manager: CorpusManager):
        """
        Initializes MorphologicalAnalysisPipeline
        """
        super().__init__(corpus_manager)
        self._backup_analyzer = pymorphy2.MorphAnalyzer()
        mapping_path = Path(__file__).parent / 'data' / 'opencorpora_tags_mapping.json'
        self._backup_tag_converter = OpenCorporaTagConverter(mapping_path)

    def _process(self, text: str) -> List[ConlluSentence]:
        """
        Returns the text representation as the list of ConlluSentence
        """
        sentences = []
        result = self._stemmer.analyze(re.sub(r'\W+', ' ', text))
        token_count = 0
        for sentence_position, sentence in enumerate(split_by_sentence(text)):
            conllu_tokens = []
            for token_position, token in enumerate(re.findall(r'\w+', sentence), start=1):
                conllu_token = ConlluToken(token)
                if not result[token_count]['text'].isalnum():
                    token_count += 1
                if 'analysis' in result[token_count] and result[token_count]['analysis']:
                    pos = self._converter.convert_pos(result[token_count]['analysis'][0]['gr'])
                    if pos == 'NOUN':
                        lex = self._backup_analyzer.parse(result[token_count]['text'])[0].normal_form
                        open_corpora_tags = self._backup_analyzer.parse(result[token_count]['text'])[0].tag
                        pos = self._backup_tag_converter.convert_pos(open_corpora_tags)
                        tags = self._backup_tag_converter.convert_morphological_tags(open_corpora_tags)
                    else:
                        tags = self._converter.convert_morphological_tags(result[token_count]['analysis'][0]['gr'])
                        lex = result[token_count]['analysis'][0]['lex']
                elif result[token_count]['text'].isdigit():
                    lex = result[token_count]['text']
                    pos = 'NUM'
                    tags = ''
                else:
                    lex = result[token_count]['text']
                    pos = 'NOUN'
                    tags = ''
                morph_params = MorphologicalTokenDTO(lex, pos, tags)
                conllu_token.set_position(token_position)
                conllu_token.set_morphological_parameters(morph_params)
                conllu_tokens.append(conllu_token)
                token_count += 1
            end_token = ConlluToken('.')
            end_token.set_position(token_position + 1)
            morph_params = MorphologicalTokenDTO('.', 'PUNCT')
            end_token.set_morphological_parameters(morph_params)
            conllu_tokens.append(end_token)
            sentence = ConlluSentence(sentence_position, sentence, conllu_tokens)
            sentences.append(sentence)
        return sentences

    def run(self) -> None:
        """
        Performs basic preprocessing and writes processed text to files
        """
        for article in self._corpus.get_articles().values():
            article.set_conllu_sentences(self._process(article.text))
            to_cleaned(article)
            to_conllu(article,
                      include_morphological_tags=True,
                      include_pymorphy_tags=True)


def main() -> None:
    """
    Entrypoint for pipeline module
    """
    manager = CorpusManager(ASSETS_PATH)
    morph_pipe = MorphologicalAnalysisPipeline(manager)
    morph_pipe.run()
    advanced_morph_pipe = AdvancedMorphologicalAnalysisPipeline(manager)
    advanced_morph_pipe.run()


if __name__ == "__main__":
    main()
