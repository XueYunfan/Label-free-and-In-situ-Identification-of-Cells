from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import utils
from tensorflow.keras.constraints import min_max_norm
from tensorflow.keras import backend as K
import tensorflow as tf

class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
 
    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')
 
        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')
 
        self.input_spec = InputSpec(ndim=ndim)
 
        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)
 
        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True
 
    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))
 
        if self.axis is not None:
            del reduction_axes[self.axis]
 
        del reduction_axes[0]
 
        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev
 
        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]
 
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed
 
    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def residual_conv_block(input_tensor, filters):
	
	norm = InstanceNormalization()(input_tensor)
	x = Activation('relu')(norm)
	x = Conv2D(filters, 3, activation = None, padding = 'same', 
		kernel_initializer = initializers.he_normal(seed=1))(x)
	x = InstanceNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(filters, 3, activation = None, padding = 'same', 
		kernel_initializer = initializers.he_normal(seed=1))(x)

	y = Conv2D(filters, 1, padding='same', activation = None, 
		kernel_initializer = initializers.he_normal(seed=1))(norm)
	
	x = add([x, y])
	
	return x

def ResUnet(input_shape=(512,512,3)):
	
	input1 = Input(input_shape)
	
	x = Conv2D(64, 3, activation = None, padding = 'same', 
		kernel_initializer = initializers.he_normal(seed=1))(input1)
	x = InstanceNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(64, 3, activation = None, padding = 'same', 
		kernel_initializer = initializers.he_normal(seed=1))(x)
	
	y = Conv2D(64, 1, padding='same', activation = None, 
		kernel_initializer = initializers.he_normal(seed=1))(input1)
	conv1 = add([x, y])
	
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	
	conv2 = residual_conv_block(pool1, 128)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	
	conv3 = residual_conv_block(pool2, 256)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	
	conv4 = residual_conv_block(pool3, 512)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
	
	conv5 = residual_conv_block(pool4, 1024)

	up1 = Conv2DTranspose(512, kernel_size=(2,2), strides=(2,2))(conv5)
	concatenate1 = concatenate([conv4,up1],axis=3)

	conv6 = residual_conv_block(concatenate1, 512)

	up2 = Conv2DTranspose(256, kernel_size=(2,2), strides=(2,2))(conv6)
	concatenate2 = concatenate([conv3,up2],axis=3)

	conv7 = residual_conv_block(concatenate2, 256)

	up3 = Conv2DTranspose(128, kernel_size=(2,2), strides=(2,2))(conv7)
	concatenate3 = concatenate([conv2,up3],axis=3)

	conv8 = residual_conv_block(concatenate3, 128)
	
	up4 = Conv2DTranspose(64, kernel_size=(2,2), strides=(2,2))(conv8)
	concatenate4 = concatenate([conv1,up4],axis=3)

	conv9 = residual_conv_block(concatenate4, 64)
	ins1 = InstanceNormalization()(conv9)
	relu1 = Activation('relu')(conv9)
	output = Conv2D(1, 1, padding='same', activation = 'sigmoid',
		kernel_initializer = initializers.he_normal(seed=1))(conv9)

	model = tf.keras.Model(inputs=input1, outputs=output)
	
	return model

if __name__ == '__main__':
	model = ResUnet()
	model.summary()
