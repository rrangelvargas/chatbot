FROM ubuntu:18.04
FROM python:3.9.5

COPY src /src
COPY requirements.txt .

RUN python3 -m pip install -r requirements.txt --no-cache-dir --progress-bar ascii

ENV PYTHONPATH "${PYTHONPATH}:/"

CMD python /src/__main__.py