'''
This is the main class that is used to read CSV files with clinical notes and to assign WHO-ICF classes and levels to each note.
The script requires a ROBERTA sentence mutilabel classification model to be downloaded and available in the specified location.
The input CSV file should have at least a column with the clinical note that is identified through the note header string.
The output copies the lines from the input CSV file and extends these with columns for the defined ICF categories.
If an ICF category and level is detected by the classifier, its value is stored in the corresponding column,

@author Jenia Kim, Piek Vossen


'''
import pandas as pd

import torch
import simpletransformers
import logging
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, classification_report
#from simpletransformers.classification import ClassificationModel, ClassificationArgs

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

def preprocess(note):
    return note.tolist()

def predict(trained_model, path_to_csvfile):
    # load dataset
    clinical_notes = pd.read_csv(path_to_csvfile, delimiter=';')
    for row in clinical_notes.rows:
        note = row["Notitietekst1"]
        if note:
            sentences = preprocess(note)
            predictions = trained_model.predict(sentences)

        else:
            print("error note header could not be matched:", note_header)
            print(clinical_notes.info())


def test_read_write_csv (path_to_csvfile):
    # input header
    # BSN;Notitie ID;NotitieCSN;Typenotitie;Notitiedatum;Zorgverlenernaam;zorgverlener specialismecode;zorgverlenertype;Notitietekst1
    bsn = "BSN"
    note_id = "Notitie ID"
    note_text = "Notitietekst1"
    date = "Notitiedatum"

    #output header
    # BSN;NotitieID;datum;FAC;ADM;ATT;BER;ENR;ETN;INS;MBW;disregard;target
    clinical_notes_input = pd.read_csv(path_to_csvfile, delimiter=';', encoding_errors='ignore')
    print(clinical_notes_input.info())
    print(clinical_notes_input.head())

    output_columns = ["BSN", "NotitieID", "datum", "FAC", "ADM", "ATT", "BER", "ENR", "ETN", "INS", "MBW", "disregard", "target"]
    clinical_notes_output = pd.DataFrame(columns=output_columns)

    for index, row in clinical_notes_input.iterrows():
        row_note_text = row[note_text]
        print(row_note_text)

        row_bsn = row[bsn]
        row_note_id = row[note_id]
        row_date = row[date]
        new_row = create_result_row(row_bsn, row_note_id, row_date)
        clinical_notes_output = clinical_notes_output.append(new_row, ignore_index=True)
        
    clinical_notes_output.to_csv(path_to_csvfile+"out.csv")

def create_result_row(row_bsn, row_note_id, row_date):
    row_fac = "FAC0-4"
    row_adm = "ADM0-3"
    row_att = "ATT0-3"
    row_ber = "BER0-3"
    row_enr = "ENR0-3"
    row_etn = "ETN0-3"
    row_ins = "INS0-3"
    row_mbw = "MBW0-3"
    row_disregard = "false"
    row_target = ""

    new_row = {"BSN": row_bsn, "NotitieID": row_note_id, "datum" : row_date,
               "FAC": row_fac, "ADM": row_adm, "ATT" : row_att, "BER": row_ber, "ENR" : row_enr, "ETN" : row_etn, "INS" : row_ins, "MBW" : row_mbw,
               "disregard" : row_disregard, "target" : row_target}
    return new_row


def main(modeltype, path_to_model, path_to_csvfile):
    print("Initialize model...")


    #classification_model = ClassificationModel(modeltype, path_to_model, use_cuda=False)
    #clinical_note_output = predict(classification_model, path_to_csvfile)

    test_read_write_csv (path_to_csvfile)

#modeltype = sys.argv[1]
#path_to_model = sys.argv[2]
#path_to_csvfile = sys.argv[3]

if __name__ == "__main__":
    modeltype = "roberta"
    path_to_model = "/Users/piek/PycharmProjects/aproof-icf-classifier/models/roberta_scratch_icf"
    path_to_csvfile = "/Users/piek/PycharmProjects/aproof-icf-classifier/example/ZorgTTP_in.csv"
    main(modeltype, path_to_model, path_to_csvfile)
