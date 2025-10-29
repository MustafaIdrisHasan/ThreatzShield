PY?=cyber_detect_backend-master/.venv/Scripts/python.exe
UVICORN?=cyber_detect_backend-master/.venv/Scripts/uvicorn.exe

.PHONY: api cli test freeze

api:
	$(UVICORN) api:app --reload

cli:
	$(PY) cli.py "Hello world"

test:
	$(PY) -m unittest discover -s tests -p "*_test.py"

freeze:
	$(PY) -m pip freeze > requirements-freeze.txt

