import argparse
import numpy as np

from model.ProtGNN import ProtGNN
from model.ProtTrans import ProtTrans


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_dim', type=int, default=2, nargs='+', help="Output dimension-> Binary/Multi-class")
    parser.add_argument('-gcd', '--gc_dims', type=int, default=[128, 128, 256], nargs='+', help="Dimensions of GraphConv layers.")
    parser.add_argument('-fcd', '--fc_dims', type=int, default=[], nargs='+', help="Dimensions of fully connected layers (after GraphConv layers).")
    parser.add_argument('-drop', '--dropout', type=float, default=0.3, help="Dropout rate.")
    parser.add_argument('-l2', '--l2_reg', type=float, default=1e-4, help="L2 regularization coefficient.")
    parser.add_argument('-lr', type=float, default=0.0002, help="Initial learning rate.")
    parser.add_argument('-gc', '--gc_layer', type=str, choices=['GraphConv', 'MultiGraphConv', 'SAGEConv', 'ChebConv', 'GAT', 'NoGraphConv'],
                        help="Graph Conv layer.")
    parser.add_argument('-e', '--epochs', type=int, default=200, help="Number of epochs to train.")
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('-pd', '--pad_len', type=int, help="Padd length (max len of protein sequences in train set).")
    parser.add_argument('-lm', '--lm_model_name', type=str, help="Path to the pretraned Language Model.")
    parser.add_argument('--cmap_type', type=str, default='ca', choices=['ca', 'cb'], help="Contact maps type.")
    parser.add_argument('--cmap_thresh', type=float, default=10.0, help="Distance cutoff for thresholding contact maps.")
    parser.add_argument('--model_name', type=str, default='GCN-PDB_MF', help="Name of the GCN model.")
    parser.add_argument('--meta_data', type=str, default="./preprocessing/data/nrPDB-GO_2019.06.18_test.csv", help="Summary File for the data")
    parser.add_argument('--train_tfrecord_fn', type=str, default="/mnt/ceph/users/vgligorijevic/ContactMaps/TFRecords/PDB_GO_train", help="Train tfrecords.")
    parser.add_argument('--valid_tfrecord_fn', type=str, default="/mnt/ceph/users/vgligorijevic/ContactMaps/TFRecords/PDB_GO_valid", help="Valid tfrecords.")

    args = parser.parse_args()
    print(args)

    train_tfrecord_fn = args.train_tfrecord_fn + '*'
    valid_tfrecord_fn = args.valid_tfrecord_fn + '*'

    output_dim = args.output_dim

    print ("### Training model: ", args.model_name, " on ", output_dim)
    model = ProtGNN(output_dim=output_dim, n_channels=26, gc_dims=args.gc_dims, fc_dims=args.fc_dims,
                    lr=args.lr, drop=args.dropout, l2_reg=args.l2_reg, gc_layer=args.gc_layer,
                    lm_model_name=args.lm_model_name, model_name_prefix=args.model_name)

    model.train(train_tfrecord_fn, valid_tfrecord_fn, epochs=args.epochs, batch_size=args.batch_size, pad_len=args.pad_len,
                cmap_type=args.cmap_type, cmap_thresh=args.cmap_thresh, class_weight=None)

    # save models
    model.save_model()
    model.plot_losses()
