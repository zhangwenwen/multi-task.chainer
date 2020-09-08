import chainer
import chainer.links as L

from chainer import initializers
import chainer.functions as F


class AttentionModule(chainer.Chain):
    def __init__(self, outchannels, initialW=None, initial_bias=None):
        super(AttentionModule, self).__init__()
        if initialW is None:
            initialW = initializers.LeCunUniform()
        if initial_bias is None:
            initial_bias = initializers.Zero()
        init = {'initialW': initialW, 'initial_bias': initial_bias}

        with self.init_scope():
            self.attention_conv1 = L.Convolution2D(outchannels, 3, stride=1, pad=1, **init)
            self.attention_conv2 = L.Convolution2D(outchannels, 3, stride=1, pad=1, **init)

    def to_gpu(self, device=None):
        super(AttentionModule, self).to_gpu(device)

    def __call__(self, first_module, second_module):
        attention = F.concat((first_module, second_module), axis=1)
        attention = self.attention_conv1(attention)
        attention = self.attention_conv2(attention)
        attention = F.sigmoid(attention)
        return attention
