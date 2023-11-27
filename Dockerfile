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

CMD ["poetry", "run", "start_fast_api"]
# CMD ["streamlit", "run", "streamlit_app/streamlit_app.py", "&", "poetry", "run", "start_fast_api"]