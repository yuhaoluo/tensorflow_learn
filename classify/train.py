# -*- coding: utf-8 -*-

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
"""
Train a CNN model.

Example Usage:
---------------
python3 train.py \
    --train_record_path: Path to training tfrecord file.
    --val_record_path: Path to validation tfrecord file.
    --model_dir: Path to log directory.
"""

import functools
import logging
import os
import tensorflow as tf

#import exporter
from model import Alexnet

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('gpu_indices', '0', 'The index of gpus to used.')
flags.DEFINE_string('train_record_path', 
                    './tf_record/train.tfrecord', 
                    'Path to training tfrecord file.')
flags.DEFINE_string('val_record_path', 
                    '/datasets/val.record', 
                    'Path to validation tfrecord file.')
flags.DEFINE_string('checkpoint_path',
                    None,
                    'Path to a pretrained model.')
flags.DEFINE_string('model_dir', './logs', 'Path to log directory.')
flags.DEFINE_float('keep_checkpoint_every_n_hours', 
                   0.2,
                   'Save model checkpoint every n hours.')
flags.DEFINE_string('learning_rate_decay_type',
                    'exponential',
                    'Specifies how the learning rate is decayed. One of '
                    '"fixed", "exponential", or "polynomial"')
flags.DEFINE_float('learning_rate', 
                   0.0001, 
                   'Initial learning rate.')
flags.DEFINE_float('end_learning_rate', 
                   0.000001,
                   'The minimal end learning rate used by a polynomial decay '
                   'learning rate.')
flags.DEFINE_float('decay_steps',
                   1000,
                   'Number of epochs after which learning rate decays. '
                   'Note: this flag counts epochs per clone but aggregates '
                   'per sync replicas. So 1.0 means that each clone will go '
                   'over full epoch individually, but replicas will go once '
                   'across all replicas.')
flags.DEFINE_float('learning_rate_decay_factor',
                   0.5,
                   'Learning rate decay factor.')
flags.DEFINE_integer('num_classes', 2, 'Number of classes.')
flags.DEFINE_integer('batch_size', 2, 'Batch size.')
flags.DEFINE_integer('num_steps', 50, 'Number of steps.')
flags.DEFINE_integer('input_size', 224, 'Number of steps.')
#flags.DEFINE_integer('steps_per_epoch', 224, 'Number of steps.')

FLAGS = flags.FLAGS

def input_fn(filenames,batch_size=1,epochs=1):
    '''
    args:
    filenames: a list of tf_record files
    '''
    def _input_fn():

        def precoss(image,label):
            image = tf.image.resize_images(image, [224, 224])
            return {'images':image}, label

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
            image = tf.image.resize_images(image, [FLAGS.input_size, FLAGS.input_size])
            image = tf.reshape(image, [FLAGS.input_size, FLAGS.input_size, 3])
            #image = transform_data(image)
            
            label = parsed["image/class/label"]
            #height = parsed['image/height']
            #width = parsed['image/width']

            return {"images":image}, label 

        datasets = tf.data.TFRecordDataset(filenames)
        datasets = datasets.map(parser)
        #datasets = datasets.map(lambda img,label:precoss(img,label))
        datasets = datasets.prefetch(batch_size)
        datasets = datasets.batch(batch_size).repeat(epochs)
        features,labels = datasets.make_one_shot_iterator().get_next() 

        return features,labels

    return _input_fn

def transform_data(image):
    #size = FLAGS.input_size + 32
    size = FLAGS.input_size 
    image = tf.squeeze(tf.image.resize_bilinear([image], size=[size, size]))
    image = tf.to_float(image)
    return image

#def create_predict_input_fn():

'''
def create_model_fn(features, labels, mode, params=None):
    params = params or {}
    loss, train_op, ... = None, None, ...

    prediction_dict = ...

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        loss = ...

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = ...

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=prediction_dict,
        loss=loss,
        train_op=train_op,
        ...)
'''
def model_fn(features,labels,mode,params=None):
    """Constructs the classification model.
    
    Modifed from:
        https://github.com/tensorflow/models/blob/master/research/
            object_detection/model_lib.py.
    
    Args:
        features: A 4-D float32 tensor with shape [batch_size, height,
            width, channels] representing a batch of images. (Support dict)
        labels: A 1-D int32 tensor with shape [batch_size] representing
             the labels of each image. (Support dict)
        mode: Mode key for tf.estimator.ModeKeys.
        params: Parameter dictionary passed from the estimator.
        
    Returns:
        An `EstimatorSpec` the encapsulates the model and its serving
        configurations.
    """  
    params = params or {}
    loss, acc, train_op, export_outputs = None, None, None, None
    is_training = (mode==tf.estimator.ModeKeys.TRAIN)

    cls_model = Alexnet(is_training=is_training,num_classes=FLAGS.num_classes)
    #preprocessed_inputs = cls_model.preprocess(features['images']) 
    preprocessed_inputs = features['images']
    prediction_dict = cls_model.predict(preprocessed_inputs)
    postprocessed_dict = cls_model.postprocess(prediction_dict)

    if mode == tf.estimator.ModeKeys.TRAIN:
        if FLAGS.checkpoint_path:
            init_variables_from_checkpoint()
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        loss_dict = cls_model.loss(postprocessed_dict,labels)
        loss = loss_dict['loss']
        pre_classes = postprocessed_dict['classes']

        acc = tf.reduce_mean(tf.cast(tf.equal(pre_classes,labels),dtype=tf.float32))
        tf.summary.scalar('loss',loss)
        tf.summary.scalar('accuracy',acc)

    scaffold = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = configure_learning_rate(FLAGS.decay_steps,global_step)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
        train_op = slim.learning.create_train_op(loss, optimizer,
                                                 summarize_gradients=True)
        keep_checkpoint_every_n_hours = FLAGS.keep_checkpoint_every_n_hours
        saver = tf.train.Saver(
                sharded=True,
                keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
                save_relative_paths=True)    
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        scaffold = tf.train.Scaffold(saver=saver)
        
        #return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op,scaffold=scaffold)

        
    eval_metric_ops = None
    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels=labels, predictions=pre_classes)
        eval_metric_ops = {'Accuracy': accuracy}

    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     export_output = exporter._add_output_tensor_nodes(postprocessed_dict)
    #     export_outputs = {
    #         tf.saved_model.signature_constants.PREDICT_METHOD_NAME:
    #             tf.estimator.export.PredictOutput(export_output)}
    return tf.estimator.EstimatorSpec(mode=mode,
                                        predictions=prediction_dict,
                                        loss=loss,
                                        train_op=train_op,
                                        eval_metric_ops=eval_metric_ops,
                                        scaffold=scaffold)


def configure_learning_rate(decay_steps, global_step):
    """Configures the learning rate.
    
    Modified from:
        https://github.com/tensorflow/models/blob/master/research/slim/
        train_image_classifier.py
    
    Args:
        decay_steps: The step to decay learning rate.
        global_step: The global_step tensor.
        
    Returns:
        A `Tensor` representing the learning rate.
    """ 
    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                         FLAGS.learning_rate_decay_type)
        
        
def init_variables_from_checkpoint(checkpoint_exclude_scopes=None):
    """Variable initialization form a given checkpoint path.
    
    Modified from:
        https://github.com/tensorflow/models/blob/master/research/
        object_detection/model_lib.py
    
    Note that the init_fn is only run when initializing the model during the 
    very first global step.
    
    Args:
        checkpoint_exclude_scopes: Comma-separated list of scopes of variables
            to exclude when restoring from a checkpoint.
    """
    exclude_patterns = None
    if checkpoint_exclude_scopes:
        exclude_patterns = [scope.strip() for scope in 
                            checkpoint_exclude_scopes.split(',')]
    variables_to_restore = tf.global_variables()
    variables_to_restore.append(slim.get_or_create_global_step())
    variables_to_init = tf.contrib.framework.filter_variables(
        variables_to_restore, exclude_patterns=exclude_patterns)
    variables_to_init_dict = {var.op.name: var for var in variables_to_init}
    
    available_var_map = get_variables_available_in_checkpoint(
        variables_to_init_dict, FLAGS.checkpoint_path, 
        include_global_step=False)
    tf.train.init_from_checkpoint(FLAGS.checkpoint_path, available_var_map)
    
    
def get_variables_available_in_checkpoint(variables,
                                          checkpoint_path,
                                          include_global_step=True):
    """Returns the subset of variables in the checkpoint.
    
    Inspects given checkpoint and returns the subset of variables that are
    available in it.
    
    Args:
        variables: A dictionary of variables to find in checkpoint.
        checkpoint_path: Path to the checkpoint to restore variables from.
        include_global_step: Whether to include `global_step` variable, if it
            exists. Default True.
            
    Returns:
        A dictionary of variables.
        
    Raises:
        ValueError: If `variables` is not a dict.
    """
    if not isinstance(variables, dict):
        raise ValueError('`variables` is expected to be a dict.')
    
    # Available variables
    ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
    ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
    if not include_global_step:
        ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)
    vars_in_ckpt = {}
    for variable_name, variable in sorted(variables.items()):
        if variable_name in ckpt_vars_to_shape_map:
            if ckpt_vars_to_shape_map[variable_name] == variable.shape.as_list():
                vars_in_ckpt[variable_name] = variable
            else:
                logging.warning('Variable [%s] is avaible in checkpoint, but '
                                'has an incompatible shape with model '
                                'variable. Checkpoint shape: [%s], model '
                                'variable shape: [%s]. This variable will not '
                                'be initialized from the checkpoint.',
                                variable_name, 
                                ckpt_vars_to_shape_map[variable_name],
                                variable.shape.as_list())
        else:
            logging.warning('Variable [%s] is not available in checkpoint',
                            variable_name)
    return vars_in_ckpt


def main(_):
    # Specify which gpu to be used
    #os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_indices
    
    estimator = tf.estimator.Estimator(model_fn=model_fn, 
                                       model_dir=FLAGS.model_dir)
    steps_per_epochs = int(10/FLAGS.batch_size)
    epochs = int(FLAGS.num_steps / steps_per_epochs)
    train_input_fn = input_fn(['tf_record/train.tfrecord'],
                              batch_size=FLAGS.batch_size,epochs=5)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                       max_steps=FLAGS.num_steps)
    eval_input_fn = input_fn(['tf_record/train.tfrecord'],
                             batch_size=FLAGS.batch_size,epochs=1)
    #predict_input_fn = create_predict_input_fn()
    #eval_exporter = tf.estimator.FinalExporter(
    # name='servo', serving_input_receiver_fn=predict_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None) #exporters=eval_exporter
    #tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    
    estimator.train(train_input_fn,max_steps=100)
    
if __name__ == '__main__':
    tf.app.run()

            
