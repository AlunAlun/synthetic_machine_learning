import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # stops tf warnings
import tensorflow as tf
import numpy as np
from numpy import array


def load_graph(frozen_graph_filename, prefix):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name=prefix)
    return graph


g = load_graph("output/frozen_model.pb", "wif")
names = [op.name for op in g.get_operations()]

#get the very first input node, should be of shape (?,3)
input_name = 'wif/X:0'
x = g.get_tensor_by_name(input_name)

#layer_7/h is the very last node of graph, shape (?, 1)
output = g.get_tensor_by_name('wif/layer_7/h:0')

print("Done.")
print("Testing with [-0.7923625, 0.0, 0.6100528], result should be [26.60093]")

# test axis
an_input = array ( [-0.7923625, 0.0, 0.6100528] )
a4d = an_input[np.newaxis] # convert to (1,3)

# #do the test by calling eval on output node
with tf.Session(graph=g) as sess:
	print( output.eval(feed_dict={x: a4d}, session=sess) ) 