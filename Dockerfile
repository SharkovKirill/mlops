FROM python:3.10.5-slim-buster as base
LABEL maintainer="kirill.office75@gmail.com"
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

WORKDIR /app

COPY poetry.lock pyproject.toml ./

RUN python -m pip install --no-cache-dir poetry==1.6.1 \
    && poetry config virtualenvs.create false \
    && poetry install --without dev --no-interaction --no-ansi \
    && rm -rf $(poetry config cache-dir)/{cache,artifacts}

COPY . /app/


ARG CLEARML_API_ACCESS_KEY="1UL30EKBNVO401UOWZG8"
ENV CLEARML_API_ACCESS_KEY="${CLEARML_API_ACCESS_KEY}"

ARG CLEARML_API_SECRET_KEY="W2twwjiEMDWdrdhfv85wV7gZESSUSamepH7yhmiQoWe2zpAWhx"
ENV CLEARML_API_SECRET_KEY="${CLEARML_API_SECRET_KEY}"

ARG CLEARML_API_HOST="http://apiserver:8008"
ENV CLEARML_API_HOST="${CLEARML_API_HOST}"

ARG CLEARML_WEB_HOST="http://web_server:8080"
ENV CLEARML_WEB_HOST="${CLEARML_WEB_HOST}"

ARG CLEARML_FILES_HOST="http://fileserver:8081"
ENV CLEARML_FILES_HOST="${CLEARML_FILES_HOST}"

ARG MINIO_USER="user_login"
ENV MINIO_USER="${MINIO_USER}"

ARG MINIO_PASSWORD="user_password"
ENV MINIO_PASSWORD="${MINIO_PASSWORD}"

ARG MINIO_URL="http://minio:9000"
ENV MINIO_URL="${MINIO_URL}"

ENV DOCKER_MODE="1"

RUN clearml-init --file clearml.conf
ENV CLEARML_CONFIG_FILE clearml.conf

CMD ["poetry", "run", "start_fast_api"]
# CMD ["streamlit", "run", "streamlit_app/streamlit_app.py", "&", "poetry", "run", "start_fast_api"]