import tensorflow as tf

'''
args:
filenames: a list of tf_record files
'''

def parser(record):
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
    #image = transform_data(image)
    image = tf.image.resize_images(image, [100, 100])
    label = parsed["image/class/label"]
    #height = parsed['image/height']
    #width = parsed['image/width']

    return (image, label)  ##{"images":image}, label


datasets = tf.data.TFRecordDataset('./tf_record/train.tfrecord')
datasets = datasets.map(parser)
datasets = datasets.batch(5).repeat(1)
features,labels = datasets.make_one_shot_iterator().get_next() 
#features,labels = datasets.prefetch(10)

with tf.Session() as sess:
    for i in range(2):
        print(sess.run(features).shape)