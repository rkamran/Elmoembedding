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


def download_imdb(data_dir, forced=False):
    print("[INFO]Downloading IMDB sentiment data...")
    file_path = os.path.sep.join([data_dir, "aclImdb_v1.tar.gz"])
    if not os.path.exists(file_path) or forced:
        request = requests.get("http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", allow_redirects=True)
        open(os.path.sep.join([data_dir, "aclImdb_v1.tar.gz"]), "wb").write(request.content)
    else:
        print("[INFO]Archive exists. Skipping download")
    
    dir_path = os.path.sep.join([data_dir, "aclImdb"])
    print("[INFO]Extracing files...")
    if not os.path.exists(dir_path) or forced:
        if os.path.exists(dir_path):
            os.removedirs(dir_path)

        tar = tarfile.open(file_path, "r:gz")
        tar.extractall(path=data_dir)
    else:
        print("[INFO]Files exist. Skipping extration")

def create_dataframe(from_dir):
    data_dict = {"sentence": [], "sentiment": [], "polarity": []}

    files = list(paths.list_files(from_dir, validExts="txt"))
    random.shuffle(files)

    print("[INFO]Processing {} files".format(len(files)))
    for file in files:        
        if file.find("/pos/") == -1  and file.find("/neg/") == -1:
            continue        
        data_dict["sentence"].append(open(file, "r", encoding='utf-8').read())
        data_dict["sentiment"].append(re.match("(.*)/(\d*)_(\d+)\.txt", file).group(3))
        data_dict["polarity"].append(0 if (file.find("/pos/") == -1) else 1)

    return pd.DataFrame(data_dict)


def process_datafram(data_frame):
    text = data_frame["sentence"].tolist()
    text = [' '.join(t.split()[:150]) for t in text]
    text = np.expand_dims(np.array(text, dtype=object), axis=1)
    label = data_frame["polarity"].values

    return (text, label)

def create_model(embedding_dim, session):
    elmo_model = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())

    elmo_embedding = lambda x: elmo_model(tf.squeeze(tf.cast(x, tf.string)), 
                                          signature="default",
                                          as_dict=True)["default"]
    
    input_text = layers.Input(shape=(1,), dtype=tf.string)
    embedding = layers.Lambda(elmo_embedding, output_shape=(embedding_dim, ))(input_text)
    dense = layers.Dense(256, activation="relu")(embedding)
    pred = layers.Dense(1, activation="sigmoid")(dense)

    model = Model(inputs=[input_text], outputs=pred)

    return model

# sess = tf.Session()
# K.set_session(sess)

# # Now instantiate the elmo model
# elmo_model = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
# sess.run(tf.global_variables_initializer())
# sess.run(tf.tables_initializer())

# print(type(elmo_model))

# elmo_model()


if __name__ == "__main__":
    download_imdb(config.PATH_DATA)
    train_text, train_label = process_datafram(create_dataframe(os.path.sep.join([config.PATH_DATA, "aclImdb", "train"])))
    test_text, test_label = process_datafram(create_dataframe(os.path.sep.join([config.PATH_DATA, "aclImdb", "test"])))

    print(train_label.shape)

    session = tf.Session()
    K.set_session(session)
    
    model = create_model(1024, session)
    model.summary()


    
    model.compile(optimizer="adam", metrics=["acc"], loss="binary_crossentropy")
    model.fit(train_text, train_label, validation_split=0.15, epochs=5, batch_size=32)


