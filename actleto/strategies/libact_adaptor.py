from .utils import MultipleQueryStrategy
from libact.base.dataset import Dataset
import numpy as np


class AdaptorLibAct:
    """Adaptor for libact query strategies."""
    
    def __init__(self, 
                 X_full_dataset,
                 y_full_dataset,
                 libact_query_alg_ctor,
                 max_samples_number = 40):
        self._train_dataset = Dataset(X_full_dataset, y_full_dataset)
        self._ctor = libact_query_alg_ctor
        self._max_samples_number= max_samples_number
        #self._train_dataset.on_update(lambda _, __ : None)
        self._train_dataset._update_callback = set()
        
    def start(self):
        self._libact_query_alg = MultipleQueryStrategy(impl=self._ctor(self._train_dataset), 
                                                       query_n=self._max_samples_number)
        self._train_dataset._update_callback = set()

    def make_iteration(self, indexes, y):
        for i in range(indexes.shape[0]):
            self._train_dataset.update(indexes[i], y[i])
        self._libact_query_alg.update(indexes, y)

    def choose_samples_for_annotation(self):
        return np.array(list(self._libact_query_alg.make_query()))


def make_libact_strategy_ctor(stg_ctor, max_samples_number):
    """Creates functor with adaptor for active learning strategies for libact."""
    def _ctor(X, y):
        return AdaptorLibAct(X, y, libact_query_alg_ctor=stg_ctor, max_samples_number=max_samples_number)
    return _ctor
