"""
Functions used in pre-processing of data for the machine learning pipelines.
"""


from src import timer


def anonymize_text(txt, nlp):
    """
    Replace entities of type PERSON and GPE with 'PERSON', 'GPE'.
    Return anonymized text.
    """
    doc = nlp(txt)
    anonym = str(doc)
    to_repl = {str(ent):ent.label_ for ent in doc.ents if ent.label_ in ['PERSON', 'GPE']}
    for string, replacement in to_repl.items():
        anonym = anonym.replace(string, replacement)
    return anonym


@timer
def anonymize(notes, nlp):
    """
    Anonymize text in pd.Series.

    Parameters
    ----------
    notes: pd.Series
        series with text
    nlp: spacy language model

    Returns
    -------
    notes: pd.Series
        series with anonymized text
    """
    print(f'Anonymizing the text in "{notes.name}". This might take a while.', flush=True)

    anonymize = lambda i: anonymize_text(i, nlp)
    return notes.apply(anonymize).rename('anonym_text')


@timer
def split_sents(notes, nlp):
    """
    Split the text in pd.Series into sentences.

    Parameters
    ----------
    notes: pd.Series
        series with text
    nlp: spacy language model

    Returns
    -------
    notes: pd.DataFrame
        df with the sentences; a column with the original note index is added
    """
    print(f'Splitting the text in "{notes.name}" to sentences. This might take a while.', flush=True)
    to_sentence = lambda txt: [str(sent) for sent in nlp(txt).sents]
    sents = notes.apply(to_sentence).explode().rename('text').reset_index().rename(columns={'index': 'note_index'})
    print(f'Done! Number of sentences: {sents.shape[0]}')
    return sents

