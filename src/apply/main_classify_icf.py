'''
This is the main class that is used to read CSV files with clinical notes and to assign WHO-ICF classes and levels to each note.
The script requires a ROBERTA sentence mutilabel classification model to be downloaded and available in the specified location.
The input CSV file should have at least a column with the clinical note that is identified through the note header string.
The output copies the lines from the input CSV file and extends these with columns for the defined ICF categories.
If an ICF category and level is detected by the classifier, its value is stored in the corresponding column,

@author Jenia Kim, Piek Vossen


'''
import os
import pandas as pd
import sys
import logging
from src import icf

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

def preprocess(note):
    return note.tolist()

def predict(trained_model, path_to_csvfile, text_column_name):
    # load dataset
    clinical_notes = pd.read_csv(path_to_csvfile, delimiter=';')
    for row in clinical_notes.rows:
        note = row[text_column_name]
        if note:
            sentences = preprocess(note)
            predictions = trained_model.predict(sentences)

        else:
            print("error note header could not be matched:", note_header)
            print(clinical_notes.info())


def test_read_write_csv (path_to_csvfile, text_column_name):

#    clinical_notes_input = pd.read_csv(path_to_csvfile, delimiter=';', encoding_errors='ignore')
    clinical_notes_input = pd.read_csv(path_to_csvfile, delimiter=';')

    clinical_notes_input[icf.output_columns] = ""

    for index, row in clinical_notes_input.iterrows():
        row_note_text = row[text_column_name]
        icf_levels = add_result_to_row(row_note_text)
        for domain in icf_levels:
            clinical_notes_input.at[index,domain] = icf_levels[domain]

    print(clinical_notes_input.info())
    print(clinical_notes_input.head(10))
    clinical_notes_input.to_csv(path_to_csvfile+"out.csv")

def add_result_to_row(row_note_text):
    print(row_note_text)
    icf_levels = {}
    for index in range(len(icf.output_columns)):
        icf_levels[icf.output_columns[index]] = icf.output_values[index]
    return icf_levels


def main(modeltype, path_to_model, path_to_csvfile, text_column_name):
    print("Initialize model...")
    #classification_model = ClassificationModel(modeltype, path_to_model, use_cuda=False)
    #clinical_note_output = predict(classification_model, path_to_csvfile)

    test_read_write_csv (path_to_csvfile, text_column_name)


if __name__ == "__main__":
    #modeltype = "roberta"
    #path_to_model = "../models/roberta_scratch_icf"
    #path_to_csvfile = "../example/input_csv.csv"
    modeltype = sys.argv[1]
    path_to_model = sys.argv[2]
    path_to_csvfile = sys.argv[3]
    text_column_name = sys.argv[4]
    if (os.path.exists(path_to_csvfile)):
        main(modeltype, path_to_model, path_to_csvfile, text_column_name)
    else:
        print('Cannot find the input CSV file at:', path_to_csvfile)
