.PHONY: install start-backend start-frontend

install:
	pip install -r requirements.txt
	cd frontend && npm install

start-backend:
	uvicorn api:app --reload --port 8000

start-frontend:
	cd frontend && npm run dev