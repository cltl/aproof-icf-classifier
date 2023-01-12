"""
The script generates predictions of the level of functioning that is described in a clinical note in Dutch. The predictions are made for 9 WHO-ICF domains: 'ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'.

The script can be customized with the following parameters:
    --in_csv: path to input csv file
    --text_col: name of the column containing the text

To change the default values of a parameter, pass it in the command line, e.g.:

$ python main.py --in_csv myfile.csv --text_col notitie_tekst
"""


import spacy
import argparse
import warnings
from pathlib import Path
from shutil import ReadError
from src.text_processing import anonymize_text
from src.icf_classifiers import load_model, predict_domains_for_sentences, predict_levels, predict_level_for_sentence
from src import timer
from statistics import mean

LEVEL_MODELS = []

@timer
def load_all_level_models(domains: []):
    print('Loading the level models upfront. This may take a while')
    for i, dom in enumerate(domains):
        level_predictions =[]
        print(f'Loading the domain-level model')
        lvl_model = load_model(
            'roberta',
            f'CLTL/icf-levels-{dom.lower()}',
            'clf',
        )
        LEVEL_MODELS.append(lvl_model)

def add_level_predictions(
    sentences,
    dom_predictions,
    domains,
):
    """
    For each domain, select the sentences in `sents` that were predicted as discussing this domain. Apply the relevant levels regression model to get level predictions and join them back to `sents`.

    Parameters
    ----------
    sents: list of sentences (text string)
    dom_predictions: list of domain predictions (list of strings) that apply to each sentence
    domains: list
        list of all the domains, in the order in which they appear in the multi-label

    Returns
    -------
    level_predictions_per_domain: list level predictions (list with floats) per domain
    """
    level_predictions_per_domain = []
    for i, dom in enumerate(domains):
        level_predictions =[]
        for sentence, dom_prediction in zip(sentences, dom_predictions):
            if dom_prediction[i]==1:
               # print(f'Generating levels predictions for {dom}.')
                level = predict_level_for_sentence(sentence, LEVEL_MODELS[i])
                level_predictions.append(level.item())
        level_predictions_per_domain.append(level_predictions)
    return level_predictions_per_domain

def process_row(row:str,
                sep: str,
                text_col_nr:int,
                nlp,
                icf_domains:[],
                domains:[]):
    labeled_row = row ### remove the newline
   # print(row)
    fields = row.split(sep)
    text = fields[text_col_nr]
    anonym_note = anonymize_text(text, nlp)
    to_sentence = lambda txt: [str(sent) for sent in nlp(txt).sents]
    sents = to_sentence(anonym_note)
    dom_predictions = predict_domains_for_sentences(sents, icf_domains)
    # predict levels
   # print('Processing domains predictions.', flush=True)
    sentence_level_predictions_per_domain = add_level_predictions(sents, dom_predictions, domains)
   # print(sentence_level_predictions_per_domain)
    #aggregate to note level
    for prediction in sentence_level_predictions_per_domain:
        if prediction:
            labeled_row+=f'{sep}{mean(prediction)}'
        else:
            labeled_row+=f'{sep}'

    labeled_row+="\n"
   # print(labeled_row)
    return labeled_row


@timer
def main(
    in_csv,
    text_col,
    sep,
    encodi):
    """
    Read the `in_csv` file, process the text by row (anonymize, split to sentences), predict domains and levels per sentence, aggregate the results back to note-level, write the results to the output file.

    Parameters
    ----------
    in_csv: str
        path to csv file with the text to process; the csv must follow the following specs: sep=';', quotechar='"', first row is the header
    text_col: str
        name of the column containing the text
    encoding: str
        encoding of the csv file, e.g. utf-8

    Returns
    -------
    None
    """

    domains=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM']

    levels = [f"{domain}_lvl" for domain in domains]

    # check path
    in_csv = Path(in_csv)
    msg = f'The csv file cannot be found in this location: "{in_csv}"'
    assert in_csv.exists(), msg

    # read csv
    print(f'Loading input csv file: {in_csv}')
    print(f'Separator: {sep}')
    print(f'Text header: {text_col}')

    in_csv_file = open(in_csv, 'r')
    ### read the headerline and check the header for the text column
    first_row = in_csv_file.readline().strip()
    headers = first_row.split(sep)
    text_column_nr = -1
    for index, header in enumerate(headers):
        print('Header', header, index)
        if header.strip()==text_col:
            text_column_nr = index
            break

    if text_column_nr ==-1:
        print(f'Could not find the text column "{text_col}" in header line: "{first_row}". Aborting.')
        return

    # text processing
    print('Loading spacy model:nl_core_news_lg')
    nlp = spacy.load('nl_core_news_lg')
    print('Loading ICF classifiers')
    # predict domain
    icf_domains = load_model(
        'roberta',
        'CLTL/icf-domains',
        'multi',
    )

    load_all_level_models(domains)
    # save output file
    out_csv = in_csv.parent / (in_csv.stem + '_output.csv')
    out_csv_file = open(out_csv, "w")
    print(f'The output will be saved to : {out_csv}')
    for level in levels:
        first_row+=sep+level
    out_csv_file.write(first_row+'\n')

    count = 0
    while True:
        count +=1
        row = in_csv_file.readline().strip()
        if not row:
            break
        else:
            labeled_row = process_row(row,sep, text_column_nr, nlp, icf_domains, domains)
            out_csv_file.write(labeled_row)
        if count%1000==0:
            print('Processed line:{}', count)

    in_csv_file.close()
    out_csv_file.close()


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in_csv', default='./example/input.csv')
    argparser.add_argument('--text_col', default='text')
    argparser.add_argument('--sep', default=';')
    argparser.add_argument('--encoding', default='utf-8')
    args = argparser.parse_args()

    main(
        args.in_csv,
        args.text_col,
        args.sep,
        args.encoding,
    )
