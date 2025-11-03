"""
Custom Keras Layers for Kinship Verification Model
===================================================
This module contains custom layers that replace Lambda layers to ensure
proper model serialization and loading.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras as keras_standalone


@keras_standalone.saving.register_keras_serializable(package="KinshipVerification")
class AbsoluteDifference(layers.Layer):
    """
    Custom layer to compute absolute difference between two tensors.
    Replaces Lambda layer for better model serialization.
    """
    
    def __init__(self, **kwargs):
        super(AbsoluteDifference, self).__init__(**kwargs)
    
    def call(self, inputs):
        """
        Compute absolute difference between two input tensors.
        
        Args:
            inputs: List of two tensors [tensor1, tensor2]
        
        Returns:
            Absolute difference: |tensor1 - tensor2|
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("AbsoluteDifference layer expects a list of 2 tensors")
        
        return tf.abs(inputs[0] - inputs[1])
    
    def get_config(self):
        """Return layer configuration for serialization."""
        config = super(AbsoluteDifference, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create layer from configuration."""
        return cls(**config)


@keras_standalone.saving.register_keras_serializable(package="KinshipVerification")
class L2Distance(layers.Layer):
    """
    Custom layer to compute L2 (Euclidean) distance between two tensors.
    """
    
    def __init__(self, **kwargs):
        super(L2Distance, self).__init__(**kwargs)
    
    def call(self, inputs):
        """
        Compute L2 distance between two input tensors.
        
        Args:
            inputs: List of two tensors [tensor1, tensor2]
        
        Returns:
            L2 distance: sqrt(sum((tensor1 - tensor2)^2))
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("L2Distance layer expects a list of 2 tensors")
        
        return tf.sqrt(tf.reduce_sum(tf.square(inputs[0] - inputs[1]), axis=-1, keepdims=True))
    
    def get_config(self):
        """Return layer configuration for serialization."""
        config = super(L2Distance, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create layer from configuration."""
        return cls(**config)


@keras_standalone.saving.register_keras_serializable(package="KinshipVerification")
class CosineSimilarity(layers.Layer):
    """
    Custom layer to compute cosine similarity between two tensors.
    """
    
    def __init__(self, **kwargs):
        super(CosineSimilarity, self).__init__(**kwargs)
    
    def call(self, inputs):
        """
        Compute cosine similarity between two input tensors.
        
        Args:
            inputs: List of two tensors [tensor1, tensor2]
        
        Returns:
            Cosine similarity: (tensor1 · tensor2) / (||tensor1|| * ||tensor2||)
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("CosineSimilarity layer expects a list of 2 tensors")
        
        # Normalize inputs
        x1_normalized = tf.nn.l2_normalize(inputs[0], axis=-1)
        x2_normalized = tf.nn.l2_normalize(inputs[1], axis=-1)
        
        # Compute cosine similarity
        return tf.reduce_sum(x1_normalized * x2_normalized, axis=-1, keepdims=True)
    
    def get_config(self):
        """Return layer configuration for serialization."""
        config = super(CosineSimilarity, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create layer from configuration."""
        return cls(**config)
