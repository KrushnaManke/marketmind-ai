.PHONY: install data pipeline app test smoke clean

install:         ## full training environment
	pip install -r requirements-train.txt

data:            ## optional: download the real datasets (needs internet)
	python -m src.download_data

pipeline:        ## data -> SQLite -> SQL -> models -> forecast -> sentiment -> figures
	python -m src.pipeline

app:
	streamlit run app/Home.py

test:
	python tests/test_project.py

smoke:           ## run every dashboard page against stubbed streamlit/plotly
	python tests/smoke_app.py

clean:
	rm -rf data/marketmind.db data/processed/* models/*.joblib reports/figures/*.png
