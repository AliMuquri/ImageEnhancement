import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, ReLU, Add, Rescaling, Layer
import numpy as np

class TileInferenceLayer(Layer):
    def __init__(self, tile_size=(256,256), stride_size=(256,256)):
        super(TileInferenceLayer,self).__init__()
        self.tile_size = tile_size
        self.stride_size = stride_size

    
    def call(self, inputs, model, training=None):       

        t_patches = tf.squeeze(tf.image.extract_patches(inputs, 
                                     sizes = [1,self.tile_size[0], self.tile_size[1], 1],
                                     strides = [1, self.stride_size[0], self.stride_size[1], 1],
                                     rates = [1,1,1,1],
                                     padding="VALID"))
       
        n_rows, n_cols, _= tf.get_static_value(tf.shape(t_patches))

        t_patches = tf.reshape(t_patches, (n_rows, n_cols,self.tile_size[0],self.tile_size[1],3))
        
        def predict_tile(tile):
            pred = model(tf.cast(tf.expand_dims(tile, axis=0), dtype=tf.float32)/255)
            return tf.squeeze(pred)
        
        pred_t = tf.stack(tf.map_fn(predict_tile, tf.reshape(t_patches, (n_rows*n_cols, self.tile_size[0], self.tile_size[1],3)), dtype=tf.float32), axis=0)
        _, new_h, new_w, channel = tf.shape(pred_t)

        final_shape = (new_h * n_rows, new_w * n_cols, 3)
        pred_t = tf.reshape(pred_t, (n_rows, n_cols, new_h, new_w, channel))
        pred_t  = tf.transpose(pred_t, [0,2,1,3,4]) 
        pred_t = tf.reshape(pred_t ,(n_cols, final_shape[0], new_w, 3))
        pred_t = tf.reshape(pred_t, (final_shape))
        output = tf.expand_dims(pred_t,0)
        return output
        

class GaussianSmoothingLayer(Layer):
    def __init__(self, kernel_size=5, sigma=2):
        super(GaussianSmoothingLayer, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self._gaussian_kernel()

    def _gaussian_kernel(self):
        gauss_x = tf.range(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1, dtype=tf.float32)
        gauss_y = tf.range(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1, dtype=tf.float32)
        gauss_x, gauss_y = tf.meshgrid(gauss_x, gauss_y)
        kernel = tf.exp(-(gauss_x ** 2 + gauss_y ** 2) / (2.0 * self.sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)  # Normalize the kernel
        kernel = tf.stack([kernel] * 3, axis=-1)  # Create a 3-channel kernel for RGB images
        kernel = tf.expand_dims(kernel, axis=-1)
        return kernel

    def call(self, inputs):
        smoothed_output = tf.nn.depthwise_conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        return smoothed_output

class ResBlock(Model):
    def __init__(self):
        super(ResBlock,self).__init__()
        self.conv1 = Conv2D(64, 3, padding="same", name="Conv1")
        self.act = ReLU(name='relu')
        self.conv2 = Conv2D(64, 3, padding="same", name="Conv2")
        self.add = Add(name='add')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.act(x)
        x = self.conv2(x)
        output = self.add([x, inputs])
        return output

class UpsamplingBlock(Model):
    def __init__(self):
        super(UpsamplingBlock, self).__init__()
        self.factor = 2
        self.conv1 = Conv2D(64 * (self.factor **2), 3, padding="same", name="Conv1")
        self.conv2 = Conv2D(64 * (self.factor **2), 3, padding="same", name="Conv2")
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = tf.nn.depth_to_space(x, self.factor)
        x = self.conv2(x)
        output = tf.nn.depth_to_space(x, self.factor)
        return output


class EDSR(Model):
    def __init__(self):
        super(EDSR,self).__init__()
        self.number_blocks = 16
        self.conv = Conv2D(64, 3, padding="same", name="Conv1")
        self.conv_blocks = [ResBlock() for _ in range(self.number_blocks)]
        self.upsamlingBlock = UpsamplingBlock()       
        self.conv2 = Conv2D(3,3, padding='same', activation=None, name='Conv2')
        self.tileinfencerlayer = TileInferenceLayer()
        self.gausslayer = GaussianSmoothingLayer()
        self.tile=False

    def call(self, inputs, tile=None, training=None):
        if self.tile:
            self.tile=False
            x = self.tileinfencerlayer(inputs, model=self)
            x = tf.clip_by_value(x ,0, 1) 
            output = self.gausslayer(x)
            return output
        
        x = self.conv(inputs)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.upsamlingBlock(x)
        output = self.conv2(x)
        return output
    
    def use_tile(self):
        self.tile=True



