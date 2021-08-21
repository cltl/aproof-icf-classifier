FROM python:3.7-slim

WORKDIR /icfc

COPY . .
RUN pip install docker-entrypoint && pip install -r ./requirements.txt 

ENTRYPOINT ["python", "./src/apply/main_classify_icf.py"]
