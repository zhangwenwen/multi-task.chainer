import chainer
import chainer.functions as F

from chainer import initializers

import chainer.links as L


class Multimask(chainer.Chain):
    def __init__(self, n_class, scale_num, initialW=None, initial_bias=None):
        self.n_class = n_class
        self.scale_num = scale_num
        super(Multimask, self).__init__()
        if initialW is None:
            initialW = initializers.LeCunUniform()
        if initial_bias is None:
            initial_bias = initializers.Zero()
        init = {'initialW': initialW, 'initial_bias': initial_bias}

        with self.init_scope():
            self.score = chainer.ChainList()
            self.upscore = chainer.ChainList()
            # self.final_score = L.Deconvolution2D(n_class, 4, stride=8, pad=0, nobias=True)
            self.final_mask = L.Convolution2D(n_class, 3, pad=1, stride=1, **init)

        for i in range(scale_num):
            self.score.add_link(L.Convolution2D(n_class, 3, pad=1))
            # self.batch_norms.add_link(L.BatchNormalization(axis=0))
        if scale_num == 6:
            self.upscore.add_link(L.Deconvolution2D(n_class, 3, stride=1, pad=0, nobias=True,
                                                    initialW=initialW))
            self.upscore.add_link(L.Deconvolution2D(n_class, 3, stride=1, pad=0, nobias=True,
                                                    initialW=initialW))
            self.upscore.add_link(L.Deconvolution2D(n_class, 2, stride=2, pad=0, nobias=True,
                                                    initialW=initialW))
            self.upscore.add_link(L.Deconvolution2D(n_class, 1, stride=2, pad=0, nobias=True,
                                                    initialW=initialW))
            self.upscore.add_link(L.Deconvolution2D(n_class, 2, stride=2, pad=0, nobias=True,
                                                    initialW=initialW))
        elif scale_num == 7:
            for i in range(scale_num - 1):
                self.upscore.add_link(L.Deconvolution2D(n_class, 2, stride=2, pad=0, nobias=True,
                                                        initialW=initialW))

    def to_gpu(self, device=None):
        super(Multimask, self).to_gpu(device)

    def __call__(self, xs):
        scores = []
        for i, x in enumerate(xs):
            # TODO:multi-mask
            score_ = self.score[i](x)

            score_ = F.relu(score_)

            scores.append(score_)
        for i, score in enumerate(scores):
            if i + 1 < len(scores):
                upscore_ = self.upscore[i](scores[i])

                scores[i + 1] = upscore_ + scores[i + 1]
        if self.scale_num == 6:
            result = chainer.functions.resize_images(scores[-1], (300, 300))
        elif self.scale_num == 7:
            result = chainer.functions.resize_images(scores[-1], (512, 512))

        del scores

        result = self.final_mask(result)

        return result
