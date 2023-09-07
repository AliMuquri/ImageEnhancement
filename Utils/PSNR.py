import tensorflow as tf

class PSNR(tf.keras.metrics.Metric):
    def __init__(self):
        super(PSNR,self).__init__(name='PSNR')
        self.max_val = 255.0
        self.psnr_sum = self.add_weight(name='psnr_sum', initializer='zeros')
        self.scale_factor = 255.0
        self.num_samples =  self.add_weight(name='num_samples', initializer='zeros')

    #update the states for each batch
    def update_state(self, y_true, y_pred, sample_weight=None):
        #reversing scaling and computing mse
        mse = tf.reduce_mean(tf.square((y_true-y_pred)*self.scale_factor), axis=[1,2,3])
        psnr =  10.0 * tf.math.log(self.max_val**2 / mse) / tf.math.log(10.0)
        
        #update the parameters
        self.psnr_sum.assign_add(tf.reduce_sum(psnr))
        self.num_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    
    #abstract method 
    def result(self):
        return self.psnr_sum/ self.num_samples

tf.keras.metrics.PSNR = PSNR
        
