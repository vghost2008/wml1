#!/usr/bin/python
import iotoolkit.save_and_read_mnist as srm
import argparse
import os
import sys
import tensorflow as tf

def main(_):
	#srm.to_records(FLAGS.data_dir,FLAGS.save_dir)
	srm.to_records("../../../mldata/fashion-mnist-org", "../../../mldata/fashion-mnist-tfdata")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str)
	parser.add_argument('--save_dir', type=str)
	FLAGS,unparsed = parser.parse_known_args()
	tf.app.run(main=main,argv=[sys.argv[0]]+unparsed)

