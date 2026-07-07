install:
	pip install -r requirements.txt
train:
	python train.py
test:
	pytest -q
