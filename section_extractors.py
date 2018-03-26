import re

def extract_impression(report_text):
    im_search = re.search('IMPRESSION:((.|\n)+?)(END OF IMPRESSION|Report dictated by:|\/\/|$)', report_text)
    if im_search:
        return im_search.group(1).lstrip().rstrip().replace(' \n', '\n')
    else:
        return ""

def extract_clinical_history(report_text):
    ch_search = re.search('CLINICAL HISTORY:((.|\n)+?)\n([A-Z]| )+:', report_text)
    if ch_search:
        return ch_search.group(1).lstrip().rstrip().replace(' \n', '\n')
    else:
        return ""

def extract_findings(report_text):
    ch_search = re.search('FINDINGS:((.|\n)+?)\n([A-Z]| )+:', report_text)
    if ch_search:
        return ch_search.group(1).lstrip().rstrip().replace(' \n', '\n')
    else:
        return ""
