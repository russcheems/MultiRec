from re import L
import tensorflow as tf

def IND_reg(weight_matrix):
    """互信息正则化
        """
    lr=0.001
    s = tf.einsum("bi,bj->ij",weight_matrix, weight_matrix)
    sum = tf.reduce_sum(s, axis=1)
    s2 = tf.einsum("bi,bi->b", weight_matrix, weight_matrix)
    return lr*tf.reduce_sum(
        -tf.math.log(s2/sum)
    )

class IndReg():
    def __init__(self,lr=0.001) -> None:
        self.lr=lr
    def __call__(self,weight_matrix):
        s = tf.einsum("bi,bj->ij",weight_matrix, weight_matrix)
        sum = tf.reduce_sum(s, axis=1)
        s2 = tf.einsum("bi,bi->b", weight_matrix, weight_matrix)
        return self.lr*tf.reduce_sum(
            -tf.math.log(s2/sum)
    )