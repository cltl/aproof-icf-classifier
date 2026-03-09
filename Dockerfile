FROM python:3.10

WORKDIR /icfc

COPY . .

RUN pip install docker-entrypoint && pip install -r ./requirements.txt && python -m spacy download nl_core_news_lg

ENTRYPOINT ["python", "./main_row_by_row.py"]
