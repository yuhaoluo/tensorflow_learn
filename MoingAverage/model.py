# -*- coding: utf-8 -*-
## Model.py
import tensorflow as tf
#from tensorflow.contrib.slim import nets
import alexnet
import tensorflow.contrib.slim as slim
#import preprocesing

class Alexnet(object):
    '''
       model definition
    '''
    def __init__(self,num_classes,is_training,default_image_size=224):
        self._num_classes = num_classes
        self._is_training = is_training
        self._default_image_size = default_image_size
        
    @property
    def num_classes(self):
        return self._num_classes
    '''
    def predict(sef,preprocessed_inputs):
        """Predict prediction tensors from inputs tensor.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.
            
        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
            net, endpoints = nets.resnet_v1.resnet_v1_50(preprocessed_inputs,
                             num_classes=None,is_training=self._is_training)
            net = tf.squeeze(net,axis=[1,2])
            logits = slim.fully_connected(net, num_outputs=self.num_classes,
                                      activation_fn=None, scope='Predict')
            prediction_dict = {'logits': logits}
        return prediction_dict
    '''
    def predict(self,preprocessed_inputs):
        """Predict prediction tensors from inputs tensor.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.
            
        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
            net, endpoints = alexnet.alexnet_v2(inputs=preprocessed_inputs,num_classes=None,
                                                is_training=self._is_training,global_pool=True)
            net = tf.squeeze(net,axis=[1,2])
            logits = slim.fully_connected(net, num_outputs=self.num_classes,
                                      activation_fn=None, scope='Predict')
            prediction_dict = {'logits': logits}
        return prediction_dict                                                           
 

    def loss(self,prediction_dict,groudtruth_list):
        """Compute scalar loss tensors with respect to provided groundtruth.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            groundtruth_lists_dict: A dict of tensors holding groundtruth
                information, with one entry for each image in the batch.
                
        Returns:
            A dictionary mapping strings (loss names) to scalar tensors
                representing loss values.
        """
        logits = prediction_dict['logits']
        tf.losses.sparse_softmax_cross_entropy(
            logits=logits, 
            labels=groudtruth_list,
            scope='Loss')
        loss = tf.losses.get_total_loss()
        loss_dict = {'loss': loss}
        return loss_dict
     
    def preprocess(self,inputs):

        return inputs

    def postprocess(self,prediction_dict):
        logits = prediction_dict['logits']
        logits = tf.nn.softmax(logits)
        pre_class = tf.argmax(logits,axis=1)
        postprocessed_dict = {'logits':logits, 'classes':pre_class}
        return  postprocessed_dict
        
    def accuracy(self,postprocessed_dict, groundtruth_lists):
        """Calculate accuracy.
        
        Args:
            postprocessed_dict: A dictionary containing the postprocessed 
                results
            groundtruth_lists: A dict of tensors holding groundtruth
                information, with one entry for each image in the batch.
                
        Returns:
            accuracy: The scalar accuracy.
        """
        pre_classes = postprocessed_dict['classes']
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pre_classes,groundtruth_lists), dtype=tf.float32))
        return accuracy
     
   