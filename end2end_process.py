from preprocessing import SectionExtractor, SentenceTokenizer, ExtraneousSentenceRemover, ReportLabeler
from semantic_mapping import DateTimeMapper, SemanticMapper, StopWordRemover, NegexSmearer, ExtenderPreserver, ExtenderRemover
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
import itertools
import pickle

def read_replacements(replacement_file_path):
    return pickle.load(open(replacement_file_path, 'rb'))

class EndToEndProcessor(TransformerMixin):
    def __init__(self, replacement_paths, sections=["impression", "findings", "clinical_history"]):
        replacements = [read_replacements(sm) for sm in replacement_paths]
        replacements = list(itertools.chain.from_iterable(replacements))
        ReplacementMapper = SemanticMapper(replacements)
        self.pipeline = make_pipeline(SectionExtractor(sections=sections),
                SentenceTokenizer(), ExtraneousSentenceRemover(), ReportLabeler(),
                DateTimeMapper, ExtenderPreserver, ReplacementMapper,
                StopWordRemover(), NegexSmearer(), ExtenderRemover, None)

    def transform(self, reports, *_):
        return self.pipeline.transform(reports)
