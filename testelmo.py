import config.sentiment_config as config
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
import argparse
import requests
import os
import tarfile
from imutils import paths
import random
import re
import pandas as pd
import numpy as np
import keras.layers as layers
from keras.models import Model
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.models import load_model


session = tf.Session()
K.set_session(session)
    
elmo_model = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
session.run(tf.global_variables_initializer())
session.run(tf.tables_initializer())

model = load_model("./elmbo_embedding.hdf5", custom_objects={"elmo_model": elmo_model, "tf":tf})
model.summary()