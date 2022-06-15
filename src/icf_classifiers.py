"""
Functions for loading models and generating predictions:

- `load_model` downloads and returns a Simple Transformers model from HuggingFace.

- `predict_domains` generates a multi-label which indicates which of the 9 ICF domains are discussed in a given sentence; the order is ['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'], i.e. if the sentence is labeled as [1, 0, 0, 0, 0, 1, 0, 0, 0], it means it contains the ADM and FAC domains

- `predict_levels` generates a float that indicates the level of functioning (for a specific domain) discussed in the sentence
"""


import numpy as np
import pandas as pd
import torch
import warnings
from simpletransformers.classification import MultiLabelClassificationModel, ClassificationModel
from src import timer


@timer
def load_model(
    model_type,
    model_name,
    task,
):
    """
    Download and return a Simple Transformers model from HuggingFace.

    Parameters
    ----------
    model_type: str
        type of the pre-trained model, e.g. bert, roberta, electra
    model_name: {str, Path}
        path to a local directory with model file or model name on Hugging Face
    task: str
        simpletransformers class: 'multi' loads MultiLabelClassificationModel, 'clf' loads ClassificationModel

    Returns
    -------
    model: MultiLabelClassificationModel or ClassificationModel
    """

    # check task
    msg = f'task should be either "multi" or "clf"; "{task}" is not valid.'
    assert task in ['multi', 'clf'], msg

    # check CUDA
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        def custom_formatwarning(msg, *args, **kwargs):
            return str(msg) + '\n'
        warnings.formatwarning = custom_formatwarning
        warnings.warn('CUDA device not available; running on a CPU!')

    # load model
    print(f'Downloading the model from https://huggingface.co/{model_name}')

    if task == 'multi':
        Model = MultiLabelClassificationModel
    else:
        Model = ClassificationModel

    return Model(
        model_type,
        model_name,
        use_cuda=cuda_available,
    )


@timer
def predict_domains(
    text,
    model,
):
    """
    Apply a fine-tuned multi-label classification model to generate predictions.

    Parameters
    ----------
    text: pd Series
        a series of strings
    model: MultiLabelClassificationModel
        fine-tuned multi-label classification model (simpletransformers)

    Returns
    -------
    df: pd Series
        a series of lists; each list is a multi-label prediction
    """

    print('Generating domains predictions. This might take a while.', flush=True)
    predictions, _ = model.predict(text.to_list())
    return pd.Series(predictions, index=text.index)


@timer
def predict_domains_for_sentences(
    sentences,
    model,
):
    """
    Apply a fine-tuned multi-label classification model to generate predictions.

    Parameters
    ----------
    text: list of sentences
    model: MultiLabelClassificationModel
        fine-tuned multi-label classification model (simpletransformers)

    Returns
    -------
    list is a multi-label prediction
    """

    print('Generating domains predictions. This might take a while.', flush=True)
    predictions, output = model.predict(sentences)
    return predictions

@timer
def predict_levels(
    text,
    model,
):
    """
    Apply a fine-tuned regression model to generate predictions.

    Parameters
    ----------
    text: pd Series
        a series of strings
    model: ClassificationModel
        fine-tuned regression model (simpletransformers)

    Returns
    -------
    predictions: pd Series
        a series of floats or an empty series (if text is empty)
    """

    to_predict = text.to_list()
    if not len(to_predict):
        return pd.Series()

    _, raw_outputs = model.predict(to_predict)
    predictions = np.squeeze(raw_outputs)
    return pd.Series(predictions, index=text.index)


@timer
def predict_level_for_sentence(
    sentence,
    model,
):
    """
    Apply a fine-tuned regression model to generate predictions.

    Parameters
    ----------
    sentence: string
    model: ClassificationModel
        fine-tuned regression model (simpletransformers)

    Returns
    -------
    list with float (if text is empty)
    """

    to_predict = [sentence]
    _, raw_outputs = model.predict(to_predict)
    predictions = np.squeeze(raw_outputs)
    return predictions