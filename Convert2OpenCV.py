import os, gc
gc.collect()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os import path
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from module.config import Config
from module.modelhelper import ModelHelper
import shutil


if __name__ == '__main__':
    config = Config()
    temp_path = path.join(config.OPENCV.CONVERT_MODE_SAVE_PATH, 'tmp', config.TIME)
    temp_file = path.join(config.OPENCV.CONVERT_MODE_SAVE_PATH, 'tmp', config.TIME, 'saved_model')

    os.makedirs(temp_path, exist_ok=True)

    modelhelper = ModelHelper(config)
    model = modelhelper.get_model()
    print("creating temp file.")
    model.save(temp_file)

    print("converting...")
    loaded = tf.saved_model.load(temp_file)
    infer = loaded.signatures['serving_default']

    f = tf.function(infer).get_concrete_function(
        input_1=tf.TensorSpec(shape=[None, config.IMG_WIDTH, config.IMG_HEIGHT, config.IMG_CHANNEL], dtype=tf.float32))
    f2 = convert_variables_to_constants_v2(f)
    graph_def = f2.graph.as_graph_def()

    # Export frozen graph
    cv_model_path = path.join(config.OPENCV.CONVERT_MODE_SAVE_PATH, config.CNN_MODEL_FILE) + '.pb'

    print("saving...")
    with tf.io.gfile.GFile(cv_model_path, 'wb') as f:
        f.write(graph_def.SerializeToString())

    print("saved!, %s" % cv_model_path)

    shutil.rmtree(temp_path, ignore_errors=True)