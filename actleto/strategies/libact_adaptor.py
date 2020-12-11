from .utils import MultipleQueryStrategy
from libact.base.dataset import Dataset
import numpy as np

import logging
logger = logging.getLogger('actleto')



class AdaptorLibAct:
    """Adaptor for libact query strategies."""
    
    def __init__(self, 
                 X_full_dataset,
                 y_full_dataset,
                 libact_query_alg_ctor):
        self._train_dataset = Dataset(X_full_dataset, y_full_dataset)
        self._ctor = libact_query_alg_ctor
        self._train_dataset._update_callback = set()
        
    def start(self):
        self._libact_query_alg = self._ctor(self._train_dataset)
        self._train_dataset._update_callback = set()

    def make_iteration(self, indexes, y):
        for i in range(indexes.shape[0]):
            self._train_dataset.update(indexes[i], y[i])
        self._libact_query_alg.update(indexes, y)

    def choose_samples_for_annotation(self):
        res = np.array(list(self._libact_query_alg.make_query()))
        return res


def make_libact_strategy_ctor(stg_ctor):
    """Creates functor with adaptor for active learning strategies for libact."""
    def _ctor(X, y):
        return AdaptorLibAct(X, y, libact_query_alg_ctor=stg_ctor)
    return _ctor
