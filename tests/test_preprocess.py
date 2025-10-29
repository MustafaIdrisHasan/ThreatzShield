import sys
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
BACKEND_DIR = BASE_DIR / "cyber_detect_backend-master"
sys.path.append(str(BACKEND_DIR))

from randomforesttest import clean


def test_clean_basic():
    text = "This, right here, is a SIMPLE test!!! 123"
    out = clean(text)
    # digits and punctuation removed; lowercase; basic stopwords dropped; stemmed
    assert "123" not in out
    assert "," not in out
    assert "this" not in out  # likely removed as a stopword (or fallback list)
    assert "simpl" in out  # stemmed form of 'simple'

