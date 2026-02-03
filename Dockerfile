FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if any (none needed for basic flask/pandas)
# RUN apt-get update && apt-get install -y ...

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the app has write permissions for the data files
RUN chmod -R 777 /app

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
