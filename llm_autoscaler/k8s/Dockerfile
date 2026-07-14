FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir flask gunicorn
COPY app.py .
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "--timeout", "30", "app:app"]
