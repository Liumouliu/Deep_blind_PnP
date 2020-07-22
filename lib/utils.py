import logging

from model.model import *

MODELS = {}

MODELS["KNNContextNormNet"] = KNNContextNormNet


def load_model(name):
  '''Creates and returns an instance of the model given its class name.
  '''
  # Find the model class from its name
  all_models = MODELS
  if name not in all_models:
    logging.warning(f'Invalid model index. You put {name}. Options are:')
    # Display a list of valid model names
    for model in all_models:
      logging.warning('\t* {}'.format(model))
    return None
  NetClass = all_models[name]

  return NetClass