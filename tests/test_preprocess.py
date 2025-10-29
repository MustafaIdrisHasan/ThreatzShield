import sys
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
BACKEND_DIR = BASE_DIR / "cyber_detect_backend-master"
sys.path.append(str(BACKEND_DIR))

import unittest
from randomforesttest import clean


class TestPreprocess(unittest.TestCase):
    def test_clean_basic(self):
        text = "This, right here, is a SIMPLE test!!! 123"
        out = clean(text)
        # digits and punctuation removed; lowercase; basic stopwords dropped; stemmed
        self.assertNotIn("123", out)
        self.assertNotIn(",", out)
        self.assertNotIn("this", out)  # likely removed as a stopword (or fallback list)
        self.assertIn("simpl", out)  # stemmed form of 'simple'


if __name__ == "__main__":
    unittest.main()
