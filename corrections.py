import csv
from sklearn.base import TransformerMixin

def read_correction_file(path):
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        while True:
            n = next(reader, None)
            if n is None:
                return
            yield (n['Report Text'], int(n['Label']))

class Corrector(TransformerMixin):
    def __init__(self, corrections):
        self.correction_map = {}
        for c in corrections:
            self.correction_map[c[0]] = c[1]
    def transform(self, reports, *_):
        result = []
        for report_obj in reports:
            if report_obj["report_text"] in self.correction_map.keys():
                report_obj["label"] = self.correction_map[report_obj["report_text"]]
            result.append(report_obj)
        return result
