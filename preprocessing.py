# Contains code for reading from CSVs, normalizing text, and labeling text
import csv
import re
from random import shuffle
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from nltk.tokenize import sent_tokenize
from section_extractors import extract_impression, extract_clinical_history, extract_findings
from corrections import Corrector, read_correction_file

def get_reports_from_zsfg_csv(corpus_path):
    with open(corpus_path) as csvfile:
        reader = csv.DictReader(csvfile)
        while True:
            n = next(reader, None)
            if n is None:
                return
            yield "IMPRESSION: " + n['Impression'].lower() + "\nEND OF IMPRESSION".replace("attending report","").replace("wet read findings have been reviewed by the attending radiologists","").replace("i agree with the above report", "").replace("with the following revisions","")

def get_reports_from_csv(corpus_path):
    with open(corpus_path) as csvfile:
        reader = csv.DictReader(csvfile)
        while True:
            n = next(reader, None)
            if n is None:
                return
            yield n['Report Text']

drop_indicators = ["is a non-reportable study", "consent form", "informed consent"]
class SectionExtractor(TransformerMixin):
    def __init__(self, sections=["impression"]):
        self.extractors =[]
        if "impression" in sections:
            self.extractors.append(extract_impression)
        if "clinical_history" in sections:
            self.extractors.append(extract_clinical_history)
        if "findings" in sections:
            self.extractors.append(extract_findings)

    def transform(self, reports, *_):
        result = []
        for report in reports:
            if True in [di in report for di in drop_indicators]:
                continue
            sections = " ".join([extractor(report) for extractor in self.extractors])
            report_obj = {"report_text" : report, "sections" : sections}
            if len([s for s in report_obj["sections"] if s != '']) > 0:
                result.append(report_obj)
        return result

punct = "!\"#$%&\'()*+,-.:;<=>?@[\]^_`{|}~\n"
class SentenceTokenizer(TransformerMixin):
    def transform(self, reports, *_):
        result = []
        for report_obj in reports:
            # Tokenize sections
            text = report_obj["sections"]
            text = text.replace("Dr.", "Dr")
            text = re.sub('[0-9]\. ', "", text)
            text = text.replace("r/o", "rule out")
            text = text.replace("R/O", "rule out")

            section_sentences = sent_tokenize(text)
            new_sentences = []
            for sentence in section_sentences:
                if len(sentence) <= 2:
                    continue
                for r in punct:
                    sentence = sentence.replace(r, " ")
                sentence = sentence.replace("/", " ")
                sentence = sentence.replace("  ", " ")
                sentence = sentence[:-1] if sentence[-1] == " " else sentence
                sentence = sentence.lower()
                new_sentences.append(sentence)
            report_obj["sentences"] = new_sentences

            # Tokenize full text
            text = report_obj["report_text"]
            text = text.replace("Dr.", "Dr")
            text = re.sub('[0-9]\. ', "", text)
            text = text.replace("r/o", "rule out")
            text = text.replace("R/O", "rule out")

            section_sentences = sent_tokenize(text)
            new_sentences = []
            for sentence in section_sentences:
                if len(sentence) <= 2:
                    continue
                for r in punct:
                    sentence = sentence.replace(r, " ")
                sentence = sentence.replace("/", " ")
                sentence = sentence.replace("  ", " ")
                sentence = sentence[:-1] if sentence[-1] == " " else sentence
                sentence = sentence.lower()
                new_sentences.append(sentence)
            report_obj["report_sentences"] = new_sentences

            result.append(report_obj)
        return result


indicator_phrases = ["discussed with", "recommendations communicated", "reported by dr",
                        "communicated with", "//impression", "results were called",
                        "communicated to", "communicated by", "called by"]
non_indicators = ["discussed with patient", "discussed with the patient", "//alert"]
def sentence_indicates_discussion(sentence):
    clean_sent = sentence
    for nip in non_indicators:
        clean_sent = clean_sent.replace(nip, "")
    return True in [ip in clean_sent for ip in indicator_phrases]

class ReportLabeler(TransformerMixin):
    def transform(self, reports, *_):
        result = []
        for report_obj in reports:
            clean_sections_sents = []
            label = 0

            for sentence in report_obj["sentences"]:
                if sentence_indicates_discussion(sentence):
                    label = 1
                else:
                    clean_sections_sents.append(sentence)

            for sentence in report_obj["report_sentences"]:
                if sentence_indicates_discussion(sentence):
                    label = 1

            if "label" in report_obj.keys():
                label = report_obj["label"]

            result.append((clean_sections_sents, label))
        return result


ext_sentence_matches = ["dictated by", "electronically signed by", "my electronic signature on this "]
class ExtraneousSentenceRemover(TransformerMixin):
    def transform(self, reports, *_):
        result = []
        for report_obj in reports:
            new_sentences = []
            for sentence in report_obj["sentences"]:
                if not True in [esm in sentence for esm in ext_sentence_matches]:
                    new_sentences.append(sentence)
            report_obj["sentences"] = new_sentences
            result.append(report_obj)
        return result

import argparse
import pickle
import itertools

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess a corpus and output it to a file')
    parser.add_argument('-i','--in_path', nargs='+', required=True)
    parser.add_argument('-c','--corrections_path', nargs='+', required=False)
    parser.add_argument('-o','--out_path', nargs=1, required=True)
    parser.add_argument('-s','--sections', nargs='+', required=True)

    args = parser.parse_args()

    data = [get_reports_from_csv(ip) if not "zsfg" in ip else get_reports_from_zsfg_csv(ip) for ip in args.in_path]
    # merged_data = list(set(list(itertools.chain.from_iterable(data))))
    merged_data = list(itertools.chain.from_iterable(data))

    corrections = [read_correction_file(cp) for cp in args.corrections_path] if args.corrections_path is not None else []
    corrections_merged = list(set(itertools.chain.from_iterable(corrections)))

    pipeline = make_pipeline(SectionExtractor(sections=args.sections), SentenceTokenizer(), ExtraneousSentenceRemover(), Corrector(corrections_merged), ReportLabeler(), None)
    preprocessed = pipeline.transform(merged_data)
    print("Writing " + str(len(preprocessed)) + " preprocessed reports")
    pickle.dump(preprocessed, open(args.out_path[0], "wb"))
