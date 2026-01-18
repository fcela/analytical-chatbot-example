.PHONY: install start-backend start-backend-legacy start-frontend

install:
	pip install -r requirements.txt
	cd frontend && npm install

start-backend:
	cd .. && python main.py

start-frontend:
	cd frontend && npm run dev