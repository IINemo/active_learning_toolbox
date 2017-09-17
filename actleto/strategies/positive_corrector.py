import pandas, numpy as np
from libact.base.dataset import Dataset, ensure_sklearn_compat
from libact.base.interfaces import ProbabilisticModel, ContinuousModel


class PositiveCorrector(object):
    """The strategy that fixes the most unprobable positive labels (from the model point of view).
    
    Note: Supports only binary classification so far.
    
    """
    
    def __init__(self, dataset, fully_annotated_y, model, strategy = 'uncertainty'):
        """
        Args:
            dataset: the libact dataset without bad annotated positive 
                labels and fully_annotated_y with positive labels.
            fully_annotated_y: the array with y labels for positives 
                and negative samples.
            model: the model for active learning.
            strategy (str): the string identifier of strategy type from 
                the list ['uncertainty', 'least_prob', 'most_prob'].
        
        """
        self.dataset = dataset
        self.dataset.on_update(self._register_update)
        
        assert isinstance(model, (ProbabilisticModel, ContinuousModel))
        self.model = model
        
        assert (strategy in ['uncertainty', 'least_prob', 'most_prob']), 'Wrong strategy identifier'
        self._strategy = strategy
        
        self._internal_dataset = pandas.DataFrame(fully_annotated_y, index = range(len(self.dataset)))
        
    def make_query(self, return_score=False):
        X, y = list(zip(*self.dataset.data))
        self.model.train(Dataset(X, self._internal_dataset.values.reshape(-1)))
        
        unlabeled_entry_ids, X_pool = list(zip(*self.dataset.get_unlabeled_entries()))
        unlabeled_entry_ids = np.asarray(unlabeled_entry_ids)
        X_pool = ensure_sklearn_compat(X_pool)

        if isinstance(self.model, ProbabilisticModel):
            score = self.model.predict_proba(X_pool)[:, 1]
        elif isinstance(self.model, ContinuousModel):
            score = self.model.predict_real(X_pool)[:, 1]
        
        if self._strategy == 'uncertainty':
            score = -np.abs(score - 0.5)
            best_id = unlabeled_entry_ids[np.argmax(score)]
        elif self._strategy == 'least_prob':
            score = -score
            best_id = unlabeled_entry_ids[np.argmax(score)]
        elif self._strategy == 'most_prob':
            best_id = unlabeled_entry_ids[np.argmax(score)]
        
        if return_score:
            return best_id, \
                   list(zip(unlabeled_entry_ids, score))
        else:
            return best_id
        
    def _register_update(self, index, answer):
        self._internal_dataset.iloc[index, 0] = answer
