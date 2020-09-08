from chainer.training import extension
from chainer.training import trigger as trigger_module


class LossSplit(extension.Extension):
    def __init__(self, trigger=(10000, 'iteration'), postprocess=None,
                 segmentation_loss_key='main/loss/mask',
                 detection_loss_loc_key='main/loss/loc', detection_loss_conf_key='main/loss/conf', smooth_alpha=0.85,
                 split_alpha=0.15):

        # conduct the action of loss division
        self._trigger = trigger_module.get_trigger(trigger)
        self.alpha = smooth_alpha
        self.split_alpha = split_alpha
        self._postprocess = postprocess
        self._segmentation_loss_key = segmentation_loss_key
        self._detection_loss_conf_key = detection_loss_conf_key
        self._detection_loss_loc_key = detection_loss_loc_key

        self._max_loss_seg = None
        self._current_loss_seg = None
        self._max_loss_det_loc = None
        self._current_loss_det_loc = None
        self._max_loss_det_conf = None
        self._current_loss_det_conf = None
        self._current_loss_split = None

    def __call__(self, trainer):
        observation = trainer.observation
        current_loss_seg = observation[self._segmentation_loss_key].data
        current_loss_det_conf = observation[self._detection_loss_conf_key].data
        current_loss_det_loc = observation[self._detection_loss_loc_key].data
        if self._max_loss_seg is None:
            self._max_loss_seg = current_loss_seg

        if self._max_loss_det_conf is None:
            self._max_loss_det_conf = current_loss_det_conf

        if self._max_loss_det_loc is None:
            self._max_loss_det_loc = current_loss_det_loc

        self._current_loss_seg = self.__smooth(self._current_loss_seg, current_loss_seg)
        self._current_loss_det_conf = self.__smooth(self._current_loss_det_conf, current_loss_det_conf)
        self._current_loss_det_loc = self.__smooth(self._current_loss_det_loc, current_loss_det_loc)

        if self._trigger(trainer):
            # compute the rewward and modify the loss split
            reward_det, reward_seg = self._reward()
            loss_split = self._loss_split(reward_det, reward_seg)
            self._current_loss_split = self.__smooth(self._current_loss_split, loss_split, self.split_alpha)

            trainer.updater._optimizers['main'].target.loss_split = self._current_loss_split

            self._max_loss_seg = current_loss_seg
            self._max_loss_det_conf = current_loss_det_conf
            self._max_loss_det_loc = current_loss_det_loc

            # trainer

    def _focal_function(self, p, r):
        '''
        -(1-p)**r*np.log(p)
        :param p:
        :param r:
        :return:
        '''
        import numpy as np
        return -(1 - p) ** r * np.log(p)

    def _reward(self):
        loss_det_current = self._current_loss_det_conf + self._current_loss_det_loc
        loss_det_max = self._max_loss_det_conf + self._max_loss_det_loc
        reward_det = (loss_det_max - loss_det_current) / loss_det_max
        reward_seg = (self._max_loss_seg - self._current_loss_seg) / self._max_loss_seg
        return reward_det, reward_seg

    def _loss_split(self, reward_det, reward_seg):
        import numpy as np
        return np.round(np.exp(reward_seg) / (np.exp(reward_seg) + np.exp(reward_det)), 4)

    def __smooth(self, previous, current, alpha=None):
        if alpha is None:
            alpha = self.alpha
        if previous is None:
            return current
        else:
            return previous * alpha + current * (1 - alpha)
