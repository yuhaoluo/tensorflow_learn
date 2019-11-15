import tensorflow as tf 

## tf 2.0

# def preprocess_image(image,resize):
#     image = tf.image.decode_jpeg(image,channels=3)
#     image = tf.image.resize(image,[resize,resize])
#     return image

# def load_and_preprocess_image(path):
#     image = tf.io.read_file(path)
#     return preprocess_image(image)

def image_input(image_path_list,label_list,batch_size,num_parallel=2):
   
    def preprocess_image(image,resize=100):
        image = tf.image.resize_images(image,[resize,resize])
        #image = tf.to_float(image)
        return image
    def load_and_preprocess_image(path,label):
        image = tf.gfile.FastGFile(path, 'rb').read()
        image = tf.image.decode_jpeg(image,channels=3)
        return preprocess_image(image),label
    
    ds = tf.data.Dataset.from_tensor_slices((image_path_list,label_list))
    ds = ds.map(load_and_preprocess_image,num_parallel_calls=num_parallel)
    ds = ds.shuffle(buffer_size=len(image_path_list))
    #ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
    ds = ds.prefetch(buffer_size=batch_size)
    ds = ds.batch(batch_size).repeat()

    images,labels = datasets.make_one_shot_iterator().get_next() 
    
    return images, labels


