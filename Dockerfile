FROM python:3.8-slim-stretch

WORKDIR /a-proof-icf-classifier

COPY . /a-proof-icf-classifier
RUN pip install -r ./requirements.txt 


ENTRYPOINT ["./src/apply/domain_classification.py]
