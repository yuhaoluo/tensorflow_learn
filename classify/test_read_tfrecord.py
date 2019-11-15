
import os
import tensorflow as tf

filenames = ["tf_record/train.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)

# Use `tf.parse_single_example()` to extract data from a `tf.Example`
# protocol buffer, and perform any additional per-record preprocessing.
def parser_fn(record):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((),tf.string,default_value=''),
        'image/format' : tf.FixedLenFeature((),tf.string,default_value='jpeg'),
        'image/class/label': tf.FixedLenFeature([],tf.int64),
        'image/height': tf.FixedLenFeature([],tf.int64),
        'image/width': tf.FixedLenFeature([],tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    image = tf.image.decode_jpeg(parsed["image/encoded"])
    #image = tf.image.resize_images(image, [100, 100])
    image = transform_data(image)
    label = parsed["image/class/label"]
 

    return {"image":image}, label

def transform_data(image):
    image = tf.squeeze(tf.image.resize_bilinear([image], size=[150, 150]))
    image = tf.to_float(image)
    return image

# Use `Dataset.map()` to build a pair of a feature dictionary and a label
# tensor for each example.
dataset = dataset.map(parser_fn)
#dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(10)
#dataset = dataset.repeat(num_epochs)

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Compute for 100 epochs.
sess = tf.Session()
with sess.as_default():
    for _ in range(1):
        sess.run(iterator.initializer)
    while True:
        try:
            data,label = sess.run(next_element)
            print(data['image'].shape)
            
        except tf.errors.OutOfRangeError:
            break
