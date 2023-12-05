# The builder image, used to build the virtual environment
FROM python:3.10-buster as builder

RUN pip install --upgrade pip 

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN touch README.md

RUN poetry install

# --without dev --no-root && rm -rf $POETRY_CACHE_DIR