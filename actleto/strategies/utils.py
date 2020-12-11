import numpy as np
from libact.base.dataset import ensure_sklearn_compat
from libact.base.interfaces import ProbabilisticModel, ContinuousModel

import logging
log = logging.getLogger('actleto')


class MultipleQueryStrategy:    
    """The helper class for quering multiple instances from the unlabeled dataset
    
    Decorator for libact strategies that queries multiple instances from unlabeled
    dataset.
    
    """
    
    def __init__(self, impl, query_n=10):
        """
        Args:
            impl: the implementation of query strategy (libact compatible).
            query_n (int): number of unannotated examples to query.
            
        """
        self.impl = impl
        self.query_n = query_n

    def make_query(self):
        try:
            id_score_list = self.impl.make_query(return_score=True)[1]
            id_score_list.sort(key = lambda p: -p[1])
            return { sample for sample, _ in id_score_list[:self.query_n] }
        except TypeError:
            return { self.impl.make_query() for _ in range(self.query_n) }
    
    def update(self, *args, **kwargs):
        return self.impl.update(*args, **kwargs)

    
class MultipleTokenQueryStrategy:    
    """The helper class for quering multiple sentences from the unlabeled dataset
    based on the number of tokens in the sentences.
    
    Decorator for libact strategies that queries multiple instances from unlabeled
    dataset.
    
    """
    
    def __init__(self, impl, n_tokens):
        """
        Args:
            impl: the implementation of query strategy (libact compatible).
            query_n (int): number of unannotated examples to query.
            
        """
        self.impl = impl
        self.n_tokens = n_tokens

    def make_query(self):
        try:
            id_score_list = self.impl.make_query(return_score=True)[1]
            id_score_list.sort(key = lambda p: -p[1])
            
            curr_index = 0
            curr_tokens = 0
            while curr_index < len(id_score_list) and curr_tokens < self.n_tokens:
                index = id_score_list[curr_index][0]
                curr_tokens += len(self.impl.dataset.data[index][0])
                curr_index += 1
            
            log.info(f'Sampled: {curr_tokens} tokens with {curr_index} sentences')
            
            return { sample for sample, _ in id_score_list[:curr_index] }
        except TypeError:
            return { self.impl.make_query() for _ in range(self.query_n) }
    
    def update(self, *args, **kwargs):
        return self.impl.update(*args, **kwargs)
    

class SklearnProbaAdapterWithUnlabeled(ProbabilisticModel):
    """The adaptor of sklearn models for libact strategies."""
    
    def __init__(self, clf):
        self._model = clf

    def train(self, dataset, *args, **kwargs):
        X, y = list(zip(*dataset.data))
        X = ensure_sklearn_compat(X)
        y = np.asarray(y)
        self._model.fit(X, y, *args, **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self._model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        X, y = list(zip(*testing_dataset.data))
        X = ensure_sklearn_compat(X)
        y = np.asarray(y)
        return self._model.score(X, y, *args, **kwargs)

    def predict_real(self, feature, *args, **kwargs):
        return self._model.predict_proba(feature, *args, **kwargs) * 2 - 1

    def predict_proba(self, feature, *args, **kwargs):
        return self._model.predict_proba(feature, *args, **kwargs)


class SklearnRealAdapter(ContinuousModel):
    """The adaptor of sklearn models for libact strategies."""
    
    def __init__(self, clf):
        self._model = clf

    def train(self, dataset, *args, **kwargs):
        self._model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self._model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self._model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)

    def predict_real(self, feature, *args, **kwargs):
        return self._model.decision_function(feature, *args, **kwargs)
