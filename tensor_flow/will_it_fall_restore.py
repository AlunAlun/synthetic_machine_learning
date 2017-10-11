'''
    File name: will_it_fall_restore.py
    Author: Alun Evabs
    Date created: 27/07/2013
    Date last modified: 27/07/2013
    Python Version: 3.5
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # stops tf warnings
import tensorflow as tf
import numpy as np
from libs import utils

sess = tf.Session()
saver = tf.train.import_meta_graph('./wif.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

#g = tf.get_default_graph()
#names = [op.name for op in g.get_operations()]
#print(names)

input_name = names[0] + ':0'
x = g.get_tensor_by_name(input_name)
print(x)
output = g.get_tensor_by_name(names[-1] + ':0')
print(output)
res = output.eval()