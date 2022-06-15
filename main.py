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
import pandas as pd
from pathlib import Path
from shutil import ReadError
from src.text_processing import anonymize, split_sents
from src.icf_classifiers import load_model, predict_domains, predict_levels
from src import timer


def add_level_predictions(
    sents,
    domains,
):
    """
    For each domain, select the sentences in `sents` that were predicted as discussing this domain. Apply the relevant levels regression model to get level predictions and join them back to `sents`.

    Parameters
    ----------
    sents: pd DataFrame
        df with sentences and `predictions` of the domains classifier
    domains: list
        list of all the domains, in the order in which they appear in the multi-label

    Returns
    -------
    sents: pd DataFrame
        the input df with additional columns containing levels predictions
    """
    for i, dom in enumerate(domains):
        boolean = sents['predictions'].apply(lambda x: bool(x[i]))
        results = sents[boolean]
        if results.empty:
            print(f'There are no sentences for which {dom} was predicted.')
        else:
            print(f'Generating levels predictions for {dom}.')
        lvl_model = load_model(
            'roberta',
            f'CLTL/icf-levels-{dom.lower()}',
            'clf',
        )
        predictions = predict_levels(results['text'], lvl_model).rename(f"{dom}_lvl")
        sents = sents.join(predictions)
    return sents


@timer
def main(
    in_csv,
    text_col,
    encoding,
):
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
    try:
        df = pd.read_csv(
            in_csv,
            sep=';',
            header=0,
            quotechar='"',
            encoding=encoding,
            low_memory=False,
        )
        print(f'Input csv file ({in_csv}) is successfuly loaded! Number of records (rows): {df.shape[0]}')
    except:
        raise ReadError('The input csv file cannot be read. Please check that it conforms with the required specifications (separator, header, quotechar, encoding).')

    # remove rows containing NA values in text column
    if df[text_col].isna().sum() > 0:
        print('Removing rows with no text:')
        print(f'Number of rows in input data: {df.shape[0]}')
        print(f'Rows containing NA in text column: {df[text_col].isna().sum()}')
        df.dropna(axis=0, subset=[text_col], inplace=True)
        print(f'Rows after dropping NA values: {df.shape[0]}')

    # text processing
    nlp = spacy.load('nl_core_news_lg')
    anonym_notes = anonymize(df[text_col], nlp)
    sents = split_sents(anonym_notes, nlp)

    # predict domains
    icf_domains = load_model(
        'roberta',
        'CLTL/icf-domains',
        'multi',
    )
    sents['predictions'] = predict_domains(sents['text'], icf_domains)

    # predict levels
    print('Processing domains predictions.', flush=True)
    sents = add_level_predictions(sents, domains)
    #print(sents.info())
    # aggregate to note-level
    note_predictions = sents.groupby('note_index')[levels].mean()
    df = df.merge(
        note_predictions,
        how='left',
        left_index=True,
        right_index=True,
    )

    # save output file
    out_csv = in_csv.parent / (in_csv.stem + '_output.csv')
    df.to_csv(out_csv, sep='\t', index=False)
    print(f'The output file is saved: {out_csv}')


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in_csv', default='./example/input.csv')
    argparser.add_argument('--text_col', default='text')
    argparser.add_argument('--encoding', default='utf-8')
    args = argparser.parse_args()

    main(
        args.in_csv,
        args.text_col,
        args.encoding,
    )
