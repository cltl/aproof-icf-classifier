'''
This is the main class that is used to read CSV files with clinical notes and to assign WHO-ICF classes and levels to each note.
The script requires a ROBERTA sentence mutilabel classification model to be downloaded and available in the specified location.
The input CSV file should have at least a column with the clinical note that is identified through the note header string.
The output copies the lines from the input CSV file and extends these with columns for the defined ICF categories.
If an ICF category and level is detected by the classifier, its value is stored in the corresponding column,

@author Jenia Kim, Piek Vossen


'''
import pandas as pd
import logging
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from preprocessing import prepareDataNC
from class_definitions import Annotation, BertContainer
from utils import lightweightDataframe, completeDataframe, filterDataframe
from eval_domain_agg import eval_per_domain
from domain_classification import make_note_df, noteLabels


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

def preprocess(note):
    return note.tolist()

def predict(trained_model, path_to_csv, note_header):
    # load dataset
    clinical_notes = pd.read_csv(path_to_csv, delimiter=';')
    for row in clinical_notes.rows:
        note = row[note_header]
        if note:
            sentences = preprocess(note)
            predictions = trained_model.predict(sentences)
    
            # select relevant columns
            clinical_note_output = clinical_notes[['sentence', 'label']]

            # turn classes into numerical classes
            clinical_note_output.loc[test_df['label'] == 'None', 'label'] = 0
            clinical_note_output.loc[test_df['label'] == '.D450: Lopen en zich verplaatsen', 'label'] = 1
            clinical_note_output.loc[test_df['label'] == '.B152: Stemming', 'label'] = 2
            clinical_note_output.loc[test_df['label'] == '.B455: Inspanningstolerantie', 'label'] = 3
            clinical_note_output.loc[test_df['label'] == '.D840-859: Beroep en werk', 'label'] = 4

            # rename columns so simpletransformers recognises them
            clinical_note_output.columns = ['text', 'labels']

            clinical_note_output['predictions'] = predictions
        else:
            print("error note header could not be matched:", note_header)
            print(clinical_notes.info())
            
    return (clinical_note_output)



def main(modeltype, path_to_model, path_to_csvfile, note_header):
    print("Initialize model...")


    classification_model = ClassificationModel(modeltype, path_to_model, note_header, use_cuda=False)
    clinical_note_output = predict(classification_model, path_to_csvfile)
    clinical_note_output


modeltype = sys.argv[1]
path_to_model = sys.argv[2]
path_to_csvfile = sys.argv[3]
note_header = sys.argv[4]

if __name__ == "__main__":
    main(modeltype, path_to_model, path_to_csvfile, note_header)
