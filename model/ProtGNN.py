import glob
import tensorflow as tf

from transformers import BertModel, BertTokenizer

from .utils import get_batched_dataset
from .layers import FuncPredictor, SumPooling
from .layers import ChebConv, GraphConv, SAGEConv, MultiGraphConv, NoGraphConv, GAT

import warnings
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class ProtGNN(object):
    """ Class containig the GCN + LM models for predicting protein function. """
    def __init__(self, output_dim, input_dim=1024, gc_dims=[64, 128], fc_dims=[512], lr=0.0002, drop=0.3, l2_reg=1e-4,
                 gc_layer=None, model_name_prefix=None):
        """ Initialize the model
        :param output_dim: {int} number of GO terms/EC numbers
        :param input_dim: {int} input dimension 
        :param gc_dims: {list <int>} number of hidden units in GConv layers
        :param fc_dims: {list <int>} number of hiddne units in Dense layers
        :param lr: {float} learning rate for Adam optimizer
        :param drop: {float} dropout fraction for Dense layers
        :param gc_layer: {str} Graph Convolution layer
        :model_name_prefix: {string} name of a ProtGNN model to be saved
        """
        self.output_dim = output_dim
        self.n_channels = n_channels
        self.model_name_prefix = model_name_prefix

        # build and compile model
        self._build_model(gc_dims, fc_dims, input_dim, output_dim, lr, drop, l2_reg, gc_layer)

    def _build_model(self, gc_dims, fc_dims, input_dim, output_dim, lr, drop, l2_reg, gc_layer=None):

        if gc_layer == 'NoGraphConv':
            self.GConv = NoGraphConv
            self.gc_layer = gc_layer
        elif gc_layer == 'GAT':
            self.GConv = GAT
            self.gc_layer = gc_layer
        elif gc_layer == 'GraphConv':
            self.GConv = GraphConv
            self.gc_layer = gc_layer
        elif gc_layer == 'MultiGraphConv':
            self.GConv = MultiGraphConv
            self.gc_layer = gc_layer
        elif gc_layer == 'SAGEConv':
            self.GConv = SAGEConv
            self.gc_layer = gc_layer
        elif gc_layer == 'ChebConv':
            self.GConv = ChebConv
            self.gc_layer = gc_layer
        else:
            self.GConv = NoGraphConv
            self.gc_layer = 'NoGraphConv'
            warnings.warn('gc_layer not specified! No GraphConv used!')

        print ("### Compiling ProtGNN model with %s layer..." % (gc_layer))

        input_cmap = tf.keras.layers.Input(shape=(None, None), name='cmap')
        input_seq = tf.keras.layers.Input(shape=(None, input_dim), name='seq')

        # Encoding layers
#         x_aa = tf.keras.layers.Dense(lm_dim, use_bias=False, name='AA_embedding')(input_seq)
#         if lm_model is not None:
#             x_lm = tf.keras.layers.Dense(lm_dim, use_bias=True, name='LM_embedding')(lm_model(input_seq))
#             x_aa = tf.keras.layers.Add(name='Embedding')([x_lm, x_aa])
#         x = tf.keras.layers.Activation('relu')(x_aa)

        # Graph Convolution layer
        gcnn_concat = []
        for l in range(0, len(gc_dims)):
            x = self.GConv(gc_dims[l], use_bias=False, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                           name=self.gc_layer + '_' + str(l+1))([input_seq, input_cmap])
            gcnn_concat.append(x)

        if len(gcnn_concat) > 1:
            x = tf.keras.layers.Concatenate(name='GCNN_concatenate')(gcnn_concat)
        else:
            x = gcnn_concat[-1]

        # Sum pooling
        x = SumPooling(axis=1, name='SumPooling')(x)

        # Dense layers
        for l in range(0, len(fc_dims)):
            x = tf.keras.layers.Dense(units=fc_dims[l], activation='relu')(x)
            x = tf.keras.layers.Dropout((l + 1)*drop)(x)

        # Output layer
        output_layer = FuncPredictor(output_dim=output_dim, name='labels')(x)

        self.model = tf.keras.Model(inputs=[input_cmap, input_seq], outputs=output_layer)
        return self.model
#         optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.95, beta_2=0.99)
#         pred_loss = tf.keras.losses.CategoricalCrossentropy()
#         self.model.compile(optimizer=optimizer, loss=pred_loss, metrics=['acc'])
#         print(self.model.summary())
