# Local shortcuts. `make data serve` builds the data and opens the site on http://localhost:8000

.PHONY: data serve test

data:
	mkdir -p public .cache
	cp -r site/. public/
	PREVIOUS_SNAPSHOT=.cache/models.json,site/data/models.json python3 -m pipeline.build_data --out public/data/models.json
	cp public/data/models.json .cache/models.json

serve:
	python3 -m http.server 8000 --directory public

test:
	python3 -m unittest discover -s tests
