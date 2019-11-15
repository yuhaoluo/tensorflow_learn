import io
import os
import tensorflow as tf
from PIL import Image


flags = tf.app.flags
flags.DEFINE_string('data_dir','data','path to images')
flags.DEFINE_string('output_path','tf_record/train.tfrecord','path to output tfrecord file')
FLAGS = flags.FLAGS

# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def create_tf_example(image_path,class_to_label):
    #with tf.gfile.GFile(image_path,'rb') as fid:
    #    encoded_jpg = fid.read()
    encoded_jpg = tf.gfile.FastGFile(image_path, 'rb').read()
    encode_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encode_jpg_io)

    width,height = image.size
    class_name = image_path.split('/')[-2]
    label = class_to_label.get(class_name)
    print(label)
    tf_example = tf.train.Example(
        features=tf.train.Features(feature={
            'image/encoded':_bytes_feature(encoded_jpg),
            'image/format':_bytes_feature('jpeg'.encode()),
            'image/class/label': _int64_feature(label),
            'image/height': _int64_feature(height),
            'image/width': _int64_feature(width)          
        })
    )
    
    return tf_example


def generate_tfrecord(image_paths,output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    class_to_label = {'cat':0,'dog':1}
    for image_file in image_paths:
        tf_example = create_tf_example(image_file,class_to_label)
        writer.write(tf_example.SerializeToString())
    writer.close()


def main(_):
    sub_dir_list = os.listdir(FLAGS.data_dir)
    #sub_dir_list = [sub_dir for sub_dir in os.listdir(FLAGS.data_dir)
    #                   if os.path.isdir(sub_dir)]
    image_paths = []
    for sub_dir in sub_dir_list:
        sub_dir = os.path.join(FLAGS.data_dir,sub_dir)
        #paths = os.listdir(sub_dir)
        paths = [os.path.join(sub_dir,image_file) for image_file in os.listdir(sub_dir) 
                                 if os.path.isfile(os.path.join(sub_dir,image_file)) ]
        image_paths += paths
    generate_tfrecord(image_paths,FLAGS.output_path)

if __name__=='__main__':
    tf.app.run()

