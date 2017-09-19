import numpy
from libact.base.interfaces import ProbabilisticModel, ContinuousModel
from libact.base.dataset import Dataset, ensure_sklearn_compat


class MPErr(object):
    """Most probable error query strategy.
    
    Finds the negative samples that are likely to be positive ones 
    from the model point of view. On each iteration, it uses the whole
    dataset for training the model: it treats unannotated samples as
    negative samples.
    
    Note: supports only binary classification so far.
    """
    
    def __init__(self, dataset, model):
        """
        Args:
            dataset: libact dataset with features.
            model: the model for active learning.
            
        """
        self.dataset = dataset
        self.model = model
        assert isinstance(self.model, (ProbabilisticModel, ContinuousModel))

    def make_query(self, return_score=False):
        X, y = list(zip(*self.dataset.data))
        self.model.train(Dataset(X,
                                 numpy.array([label if not label is None else False
                                 for label in y])))

        unlabeled_entry_ids, X_pool = list(zip(*self.dataset.get_unlabeled_entries()))
        unlabeled_entry_ids = numpy.asarray(unlabeled_entry_ids)
        X_pool = ensure_sklearn_compat(X_pool)

        if isinstance(self.model, ProbabilisticModel):
            score = self.model.predict_proba(X_pool)[:, 1]
        elif isinstance(self.model, ContinuousModel):
            score = self.model.predict_real(X_pool)[:, 1]
        
        best_id = unlabeled_entry_ids[numpy.argmax(score)]
        if return_score:
            return best_id, \
                   list(zip(unlabeled_entry_ids, score))
        else:
            return best_id
        