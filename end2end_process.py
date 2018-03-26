from preprocessing import SectionExtractor, SentenceTokenizer, ExtraneousSentenceRemover, ReportLabeler
from semantic_mapping import DateTimeMapper, SemanticMapper, StopWordRemover, NegexSmearer, ExtenderPreserver, ExtenderRemover
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
import pickle

def read_replacements(replacement_file_path):
    return pickle.load(open(replacement_file_path, 'rb'))

class EndToEndProcessor(TransformerMixin):
    def __init__(self, replacement_file_path, radlex=None, sections=["impression", "findings", "clinical_history"]):
        replacements = read_replacements(replacement_file_path)
        ReplacementMapper = SemanticMapper(replacements)
        if radlex is None:
            self.pipeline = make_pipeline(SectionExtractor(sections=sections),
                SentenceTokenizer(), ExtraneousSentenceRemover(), ReportLabeler(),
                ExtenderPreserver, ReplacementMapper, DateTimeMapper,
                StopWordRemover(), NegexSmearer(), ExtenderRemover, None)
        else:
            radlex_replacements = read_replacements(radlex)
            RadlexMapper = SemanticMapper(radlex_replacements)
            self.pipeline = make_pipeline(SectionExtractor(sections=sections),
                SentenceTokenizer(), ExtraneousSentenceRemover(), ReportLabeler(),
                ExtenderPreserver, ReplacementMapper, RadlexMapper, DateTimeMapper,
                StopWordRemover(), NegexSmearer(), ExtenderRemover, None)

    def transform(self, reports, *_):
        return self.pipeline.transform(reports)
