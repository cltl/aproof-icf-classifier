a-proof-icf-classifier
=============
# Contents
1. [Description](#description)
2. [Input File](#input-file)
3. [Output File](#output-file)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [How to use?](#how-to-use)
6. [Cached models](#cached-models)
7. [Runtime and File Size](#runtime-and-file-size)

# Description
This repository contains a machine learning pipeline that reads a clinical note in Dutch and assigns the functioning level of the patient based on the textual description.

We focus on 9 [WHO-ICF](https://www.who.int/standards/classifications/international-classification-of-functioning-disability-and-health) domains, which were chosen due to their relevance to recovery from COVID-19:

ICF code | Domain | name in repo
---|---|---
b1300 | Energy level | ENR
b140 | Attention functions | ATT
b152 | Emotional functions | STM
b440 | Respiration functions | ADM
b455 | Exercise tolerance functions | INS
b530 | Weight maintenance functions | MBW
d450 | Walking | FAC
d550 | Eating | ETN
d840-d859 | Work and employment | BER

### Functioning Levels
- FAC and INS have a scale of 0-5, where 5 means there is no functioning problem.
- The rest of the domains have a scale of 0-4, where 4 means there is no functioning problem.
- For more information about the levels, refer to the [annotation guidelines](https://github.com/cltl/a-proof-zonmw/tree/main/resources/annotation_guidelines).
- **NOTE**: the values generated by the machine learning pipeline might sometimes be outside of the scale (e.g. 4.2 for ENR); this is normal in a regression model.

# Input file
The input is a csv file with at least one column containing the text (one clinical note per row).

The csv must follow the following specifications:
- sep = ;
- quotechar = "
- the first row is the header (column names)

See example in [example/input.csv](example/input.csv).

# Output file
The output file is saved in the same location as the input; it has 'output' added to the original file name.

The output file contains the same columns as the input + 9 new columns with the functioning levels per domain.

The functioning levels are generated per row. If a cell is empty, it means that this domain is not discussed in this note (according to the algorithm).

See example in [example/input_output.csv](example/input_output.csv).

# Machine Learning Pipeline
The pipeline includes a multi-label classification model that detects the domains mentioned in a sentence, and 9 regression models that assign a level to sentences in which a specific domain was detected. All models were created by fine-tuning a pre-trained [Dutch medical language model](https://github.com/cltl-students/verkijk_stella_rma_thesis_dutch_medical_langauge_model).

The pipeline includes the following steps:

![ml_pipe drawio](https://user-images.githubusercontent.com/38586487/134154846-32c38fe2-e9c9-4831-962c-c180b39e6928.png)

# How to use?
## Step 1: Setting up Docker
1. Install Docker Desktop: see [here](https://docs.docker.com/desktop/windows/install/) for Windows and [here](https://docs.docker.com/desktop/mac/install/) for macOS.
2. Pull the docker image from [DockerHub](https://hub.docker.com/r/piekvossen/a-proof-icf-classifier) by typing in your command line:
```bash
docker pull piekvossen/a-proof-icf-classifier
```
3. Run the docker on the [example/input.csv](example/input.csv) file (it is already in the docker image and is given as the default argument to the [main.py](main.py) script):
```bash
docker run piekvossen/a-proof-icf-classifier
```
This will download all the required models from [https://huggingface.co/CLTL](https://huggingface.co/CLTL) and store them in the Docker's `.cache`, so that in subsequent runs cached models can be used. In total, 10 transformers models are downloaded, each between 500MB and 1GB.

## Step 2: Running the pipeline on your data
To run the pipeline on your own data (i.e. a csv file on your local machine), you need to mount the local directory where the file is stored to the docker container. This is done with the `-v` flag and then `<local_dir>:<docker_dir>`. In addition, you need to pass the following arguments:
- `--in_csv`: path to the input csv file
- `--text_col`: name of the text column in the csv
- `--sep`: separator character that separates the columns in the csv
- `--encoding` (optional): use if input csv is not utf-8

For example, if your csv file is in `C:\Users\User\Desktop`, it is called `myfile.csv` and the text is in the column `note` where columns are seprated with ";" you need to run the following command:
```bash
docker run -v C:\Users\User\Desktop:/root piekvossen/a-proof-icf-classifier --in_csv /root/myfile.csv --text_col note --sep ';'
```

# Cached models
To save the cached models on the local file system, or use them in a different container in a follow-up run, mount the Huggingface cache dir to a local directory. For example:
```bash
docker run -v <local_path_to_cache>:/root/.cache/huggingface/transformers/ piekvossen/a-proof-icf-classifier --in_csv example/input.csv --text_col text --sep ';'
```

To use the cached models in an environment without internet connection, set `TRANSFORMERS_OFFLINE=1` as environment variable (see [Huggingface documentation](https://huggingface.co/transformers/installation.html#offline-mode)). For example:

```bash
docker run -v <local_path_to_cache>:/root/.cache/huggingface/transformers/ -e TRANSFORMERS_OFFLINE=1 piekvossen/a-proof-icf-classifier --in_csv example/input.csv --text_col text --sep ';'
```

# Runtime and File Size
The code runs faster if GPU is available on your machine; it is used automatically if it's available, no need to configure anything.

On some machines, you might run into issues when generating domains predictions (this function is applied to each sentence in the input file). If this is the case, split the input into smaller batches.

# Reference

When using this repository please cite:

J. Kim, S. Verkijk, E. Geleijn, M. van der Leeden, C. Meskers, C. Meskers, S. van der Veen, P. Vossen, and G. Widdershoven, Modeling dutch medical texts for detecting functional categories and levels of covid-19 patients, 2022. 

## Bibtext:

@proceedings{kim-etal-lrec2022,
author={Jenia Kim and Stella Verkijk and Edwin Geleijn and Marieke van der Leeden and Carel Meskers and Caroline Meskers and Sabina van der Veen and Piek Vossen and Guy Widdershoven},
title={Modeling Dutch Medical Texts for Detecting Functional Categories and Levels of COVID-19 Patients},
booktitle={Proceedings of the 13th Language Resources and Evaluation Conference, Marseille, June, 2022},
year={2022}
}

