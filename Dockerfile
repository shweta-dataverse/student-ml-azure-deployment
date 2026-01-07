FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y build-essential \
    && pip install --no-cache-dir -r requirements.txt

# Expose port 80
EXPOSE 80
CMD ["python3", "app.py"]   
