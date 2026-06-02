.PHONY: run run-local-backend run-local-frontend stop

# Default command to start everything using Docker
run:
	docker-compose up --build

# Stop the Docker containers
stop:
	docker-compose down

# Alternative: run backend locally (without Docker)
run-local-backend:
	uvicorn main:api --host 0.0.0.0 --port 8000 --reload

# Alternative: run frontend locally (without Docker)
run-local-frontend:
	streamlit run app.py
