FROM python:3.9-slim

WORKDIR /src

RUN apt-get update && \
    apt-get install -y postgresql-client libpq-dev build-essential

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install flask-caching

# Copy all necessary files from src directory
COPY src/ /src/

ENV FLASK_APP=app.py
ENV PYTHONPATH=/src

CMD ["flask", "run", "--host=0.0.0.0"]