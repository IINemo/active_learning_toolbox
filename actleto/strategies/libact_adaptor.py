from .utils import MultipleQueryStrategy
from libact.base.dataset import Dataset
import pandas as pd
import numpy as np


class AdaptorLibAct(object):
    """Adaptor for libact query strategies."""
    
    def __init__(self, 
                 X_full_dataset,
                 y_full_dataset,
                 libact_query_alg_ctor,
                 max_samples_number = 40):
        super().__init__()
        
        notnull = pd.Series(y_full_dataset, index = list(range(y_full_dataset.shape[0])))
        notnull = notnull[notnull.notnull()]
        self._train_dataset = Dataset(X_full_dataset, y_full_dataset)
        self._libact_query_alg = MultipleQueryStrategy(impl = libact_query_alg_ctor(self._train_dataset), 
                                                       query_n = max_samples_number)

    def make_iteration(self, indexes, y):
        for i in range(indexes.shape[0]):
            self._train_dataset.update(indexes[i], y[i])

    def choose_samples_for_annotation(self):
        return np.array(list(self._libact_query_alg.make_query()))


def make_libact_strategy_ctor(stg_ctor):
    """Creates functor with adaptor for active learning strategies for libact."""
    return lambda X, y : AdaptorLibAct(X, y, libact_query_alg_ctor = stg_ctor)
