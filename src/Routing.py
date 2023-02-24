import regex
import tensorflow as tf
from tensorflow.keras.layers import Layer
class HardShaedLayer(Layer):
    def __init__(self,shape,reg,name=None, **kwargs) -> None:
        # self.name=name
        super(HardShaedLayer,self).__init__(**kwargs)
        self.shape=shape
        self.reg=reg
    def build(self,input_shape):
        if self.reg==None:
            self.kernel = self.add_weight("kernel", self.shape)
        else:
            self.kernel = self.add_weight("kernel", self.shape,regularizer=self.reg)
    def call(self,input):
        a=tf.matmul(input,tf.transpose(self.kernel,[1,0]))
        # a=tf.einsum("bi,ji->bj",input,self.kernel)
        b=tf.nn.softmax(a,axis=1)
        idx=tf.argmax(b,axis=1)
        out=tf.nn.embedding_lookup(self.kernel,idx)
        # out=tf.reduce_sum(tf.exp(tf.matmul(input,self.kernel)),axis=1) # 点乘
        # out=tf.nn.softmax(out)
        # idx=tf.argmax(out)
        # out=tf.nn.embedding_lookup(self.kernel,idx)
        return out


class SoftShaedLayer(Layer):
    def __init__(self,shape,reg=None,name=None, **kwargs) -> None:
        # self.name=name
        super(SoftShaedLayer,self).__init__(**kwargs)
        self.shape=shape
        self.reg=reg
    def build(self,input_shape):
        if self.reg==None:
            self.kernel = self.add_weight("kernel", self.shape)
        else:
            self.kernel = self.add_weight("kernel", self.shape,regularizer=self.reg)
    def call(self,input):
        a=tf.matmul(input,tf.transpose(self.kernel,[1,0]))
        # a=tf.einsum("bi,ji->bj",input,self.kernel)
        b=tf.nn.softmax(a,axis=1)
        b=tf.tile(tf.expand_dims(b,-1),[1,1,self.kernel.shape[1]])
        out=tf.einsum("ij,bij->bj",self.kernel,b)
        return out



def build_shared_layer(mode,shape,reg=None,name=None, **kwargs):
    if mode.lower()=="hard":
        return HardShaedLayer(shape,reg,name,**kwargs)
    elif mode.lower()=="soft":
        return SoftShaedLayer(shape,reg,name,**kwargs)
    else:
        raise "do not support {} mode, please choice one of {hard, soft}".format(mode)