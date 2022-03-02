import csv
import glob
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score

from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser


def load_predicted_PDB(pdbfile):
    # Generate (diagonalized) C_alpha distance matrix from a pdbfile
    parser = PDBParser()
    structure = parser.get_structure(pdbfile.split('/')[-1].split('.')[0], pdbfile)
    residues = [r for r in structure.get_residues()]

    # sequence from atom lines
    records = SeqIO.parse(pdbfile, 'pdb-atom')
    seqs = [str(r.seq) for r in records]

    distances = np.empty((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord()
            distances[x, y] = np.linalg.norm(one-two)

    return distances, seqs[0]


def norm_adj(A, symm=True):
    #  Normalize adj matrix
    A += np.eye(A.shape[1])
    if symm:
        d = 1.0/np.sqrt(A.sum(axis=1))
        D = np.diag(d)
        A = D.dot(A.dot(D))
    else:
        A /= A.sum(axis=1)[:, np.newaxis]

    return A


def _micro_aupr(y_true, y_test):
    return average_precision_score(y_true, y_test, average='micro')


def micro_aupr(y_true, y_pred):
    return tf.py_func(_micro_aupr, (y_true, y_pred), tf.double)


def seq2onehot(seq):
    """Create 26-dim embedding"""
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))

    # Convert vocab to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    embed_x = [vocab_embed[v] for v in seq]
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])

    return seqs_x


def _parse_function_gcn(serialized, n_goterms, channels=26, cmap_type='ca', cmap_thresh=10.0, ont='mf'):
    features = {
        cmap_type + '_dist_matrix': tf.io.VarLenFeature(dtype=tf.float32),
        "seq_1hot": tf.io.VarLenFeature(dtype=tf.float32),
        ont + "_labels": tf.io.FixedLenFeature([n_goterms], dtype=tf.int64),
        "L": tf.io.FixedLenFeature([1], dtype=tf.int64)
    }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)

    # Get all data
    L = parsed_example['L'][0]

    A_shape = tf.stack([L, L])
    A = parsed_example[cmap_type + '_dist_matrix']
    A = tf.cast(A, tf.float32)
    A = tf.sparse.to_dense(A)
    A = tf.reshape(A, A_shape)

    # threshold distances
    A_cmap = tf.cast(tf.less_equal(A, cmap_thresh), tf.float32)

    S_shape = tf.stack([L, channels])
    S = parsed_example['seq_1hot']
    S = tf.cast(S, tf.float32)
    S = tf.sparse.to_dense(S)
    S = tf.reshape(S, S_shape)

    labels = parsed_example[ont + '_labels']
    labels = tf.cast(labels, tf.float32)

    inverse_labels = tf.cast(tf.equal(labels, 0), dtype=tf.float32)  # [batch, classes]
    y = tf.stack([labels, inverse_labels], axis=-1)  # labels, inverse labels
    y = tf.reshape(y, shape=[n_goterms, 2])  # [batch, classes, Pos-Neg].

    return {'cmap': A_cmap, 'seq': S}, y


def _parse_function_cnn(serialized, n_goterms, channels=26, ont='mf'):
    features = {
        "seq_1hot": tf.io.VarLenFeature(dtype=tf.float32),
        ont + "_labels": tf.io.FixedLenFeature([n_goterms], dtype=tf.int64),
        "L": tf.io.FixedLenFeature([1], dtype=tf.int64),
    }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)

    # Get all data
    L = parsed_example['L'][0]

    S_shape = tf.stack([L, channels])
    S = parsed_example['seq_1hot']
    S = tf.cast(S, tf.float32)
    S = tf.sparse.to_dense(S)
    S = tf.reshape(S, S_shape)

    labels = parsed_example[ont + '_labels']
    labels = tf.cast(labels, tf.float32)

    inverse_labels = tf.cast(tf.equal(labels, 0), dtype=tf.float32)  # [batch, classes]
    y = tf.stack([labels, inverse_labels], axis=-1)  # labels, inverse labels
    y = tf.reshape(y, shape=[n_goterms, 2])  # [batch, classes, Pos-Neg].

    return S, y


def get_batched_dataset(filenames, batch_size=64, pad_len=1000, n_goterms=347, channels=26, cmap_type='ca', cmap_thresh=10.0):
    # settings to read from all the shards in parallel
    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    # list all files
    filenames = tf.io.gfile.glob(filenames)
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)

    # Parse the serialized data in the TFRecords files.
    dataset = dataset.map(lambda x: _parse_function_gcn(x, n_goterms=n_goterms, channels=channels, cmap_type=cmap_type, cmap_thresh=cmap_thresh))

    # Randomizes input using a window of 2000 elements (read into memory)
    dataset = dataset.shuffle(buffer_size=2000 + 3*batch_size)

    dataset = dataset.padded_batch(batch_size, padded_shapes=({'cmap': [pad_len, pad_len], 'seq': [pad_len, channels]}, [None, 2]))
    # dataset = dataset.padded_batch(batch_size, padded_shapes=({'cmap': [pad_len, pad_len], 'seq': [pad_len, channels]}, [None]))
        
    dataset = dataset.repeat()

    return dataset


def load_catalogue(fn='/mnt/home/dberenberg/ceph/SWISSMODEL_CONTACTMAPS/catalogue.csv'):
    chain2path = {}
    with open(fn) as tsvfile:
        fRead = csv.reader(tsvfile, delimiter=',')
        # next(fRead, None)
        for line in fRead:
            pdb_chain = line[0].strip()
            path = line[1].strip()
            chain2path[pdb_chain] = path
    return chain2path


if __name__ == "__main__":

    filenames = '/mnt/ceph/users/vgligorijevic/ContactMaps/TFRecords/PDB_GO_valid*'
    n_records = sum(1 for f in glob.glob(filenames) for _ in tf.data.TFRecordDataset(f))
    print ("### Total number of samples=", n_records)
