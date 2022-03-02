import glob
import tensorflow as tf

from utils.utils import get_batched_dataset
from model.layers import FuncPredictor, SumPooling

import warnings
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def train(model, train_tfrecord_fn, valid_tfrecord_fn,
          epochs=100, batch_size=64, pad_len=1200, cmap_type='ca', cmap_thresh=10.0, ont='mf', class_weight=None):

    n_train_records = sum(1 for f in glob.glob(train_tfrecord_fn) for _ in tf.data.TFRecordDataset(f))
    n_valid_records = sum(1 for f in glob.glob(valid_tfrecord_fn) for _ in tf.data.TFRecordDataset(f))
    print ("### Training on: ", n_train_records, "contact maps.")
    print ("### Validating on: ", n_valid_records, "contact maps.")

    # train tfrecords
    batch_train = get_batched_dataset(train_tfrecord_fn,
                                      batch_size=batch_size,
                                      pad_len=pad_len,
                                      n_goterms=self.output_dim,
                                      channels=self.n_channels,
                                      cmap_type=cmap_type,
                                      cmap_thresh=cmap_thresh,
                                      ont=ont)

    # validation tfrecords
    batch_valid = get_batched_dataset(valid_tfrecord_fn,
                                      batch_size=batch_size,
                                      pad_len=pad_len,
                                      n_goterms=self.output_dim,
                                      channels=self.n_channels,
                                      cmap_type=cmap_type,
                                      cmap_thresh=cmap_thresh,
                                      ont=ont)

    # early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    # model checkpoint
    mc = tf.keras.callbacks.ModelCheckpoint(self.model_name_prefix + '_best_train_model.h5', monitor='val_loss', mode='min', verbose=1,
                                            save_best_only=True, save_weights_only=True)

    # fit model
    history = model.fit(batch_train,
                        epochs=epochs,
                        validation_data=batch_valid,
                        steps_per_epoch=n_train_records//batch_size,
                        validation_steps=n_valid_records//batch_size,
                        class_weight=class_weight,
                        callbacks=[es, mc])

    history = history.history

def predict(input_data):
    return model(input_data).numpy()[0][:, 0]

def plot_losses():
    plt.figure()
    plt.plot(self.history['loss'], '-')
    plt.plot(self.history['val_loss'], '-')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(self.model_name_prefix + '_model_loss.png', bbox_inches='tight')

    plt.figure()
    plt.plot(self.history['acc'], '-')
    plt.plot(self.history['val_acc'], '-')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(self.model_name_prefix + '_model_accuracy.png', bbox_inches='tight')

def save_model(model, model_name_prefix):
    model.save(model_name_prefix + '.hdf5')

def load_model(model_name_prefix):
    model = tf.keras.models.load_model(self.model_name_prefix + '.hdf5',
                                            custom_objects={self.gc_layer: self.GConv,
                                                            'FuncPredictor': FuncPredictor,
                                                            'SumPooling': SumPooling
                                                            })
    return model
