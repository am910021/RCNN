import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from module.config import Config
from module.modelhelper import ModelHelper


if __name__ == '__main__':
   config = Config()
   modelhelper = ModelHelper(config)

   f = tf.function(modelhelper.get_model()).get_concrete_function(input_1=tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32))
   f2 = convert_variables_to_constants_v2(f)
   graph_def = f2.graph.as_graph_def()

   # Export frozen graph
   with tf.io.gfile.GFile('frozen_graph.pb', 'wb') as f:
      f.write(graph_def.SerializeToString())