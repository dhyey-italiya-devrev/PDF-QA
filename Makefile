.PHONY: clean

clean:
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pkl" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.faiss" -delete
	find . -type f -name "*.index" -delete

.DEFAULT_GOAL := clean
