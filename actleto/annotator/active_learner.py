import numpy as np
import pandas as pd
import logging

# TODO: partial fit.
# TODO: separate thread for evaluation.
# TODO: separate thread for training al model.

logger = logging.getLogger('actleto')


class ActiveLearner(object):
    """The class that implements active learning logic."""
    
    def __init__(self, 
                 active_learn_alg_ctor,
                 X_full_dataset, 
                 y_dtype,
                 y_full_dataset = None, 
                 model_evaluate = None,
                 X_test_dataset = None,
                 y_test_dataset = None,
                 eval_metrics = None,
                 rnd_start_steps = 0,
                 rnd_start_samples = 10): 
        """ActiveLearner constructor.
        
        Args:
            active_learn_alg_ctor (functor): functor object that returns active learning strategy.
            X_full_dataset (np.array or sparse matrix): feature matrix.
            y_dtype: type of y labels.
            y_full_dataset (np.array): known answers (e.g., None -- unknown, True -- positive class, False -- negative class) 
            model_evaluate: the model that will be evaluated on the holdout.
            X_test_dataset: feature matrix for testing via holdout.
            y_test_dataset: y labels for testing via holdout.
            eval_metrics (list): list of sklearn evaluation metrics.
            rnd_start_steps: AL will can make several seed steps by choosing random samples (without model suggestions).
            logger (logging.Logger): the object for logging.
            
        """
        super().__init__()
        
        self._y_dtype = y_dtype
        self._model_evaluate = model_evaluate
        self._eval_metrics = eval_metrics
        
        self._X_full_dataset = X_full_dataset
        if y_full_dataset is not None:
            self._y_full_dataset = y_full_dataset # TODO: validate dimentions
        else:
            self._y_full_dataset = [None] * self._X_full_dataset.shape[0]
            
        self._active_learn_algorithm = active_learn_alg_ctor(self._X_full_dataset, 
                                                             self._y_full_dataset)
        
        self._X_test_dataset = X_test_dataset
        self._y_test_dataset = y_test_dataset
        
        self._iteration_num = 0
        self._rnd_start_steps = rnd_start_steps
        self._rnd_start_samples = rnd_start_samples

    def _select_unannotated(self, labels):
        return np.where([(e is None) for e in labels])[0]
    
    def start(self):
        self._active_learn_algorithm.start()

    def choose_random_sample_for_annotation(self, number):
        return np.random.choice(self._select_unannotated(self._y_full_dataset), 
                                size = number, 
                                replace = False)
        
    def choose_samples_for_annotation(self):
        if self._iteration_num < self._rnd_start_steps:
            return self.choose_random_sample_for_annotation(self._rnd_start_samples)
        else:
            return self._active_learn_algorithm.choose_samples_for_annotation()
    
    def evaluate(self, fit_model=True):
        if self._model_evaluate is None:
            return None
        
        if fit_model:
            selector = [n for n, _ in enumerate(self._y_full_dataset) if e is not None]
            y_fit = [self._y_full_dataset[i] for i in selector]
            #y_fit = [e for e in self._y_full_dataset if e is not None]

            #y_fit = pd.Series(self._y_full_dataset)
            #y_fit = y_fit[y_fit.notnull()].astype(self._y_dtype)
            logger.info('Number of training samples: {}'.format(len(y_fit)))

            self._model_evaluate.fit(self._X_full_dataset[selector], y_fit)
        
        preds = self._model_evaluate.predict(self._X_test_dataset)
        return {metric.__name__ : metric(preds, self._y_test_dataset) 
                for metric in self._eval_metrics}
    
    def get_annotation(self):
        return self._y_full_dataset
    
    def make_iteration(self, indexes, answers):
        self._iteration_num += 1
        
        selector = ((answers == np.array(None)).sum(axis=1) == 0)
        answers = answers[selector]
        indexes = indexes[selector]
        for num, i in enumerate(indexes):
            self._y_full_dataset[i] = answers[num]
        
        return self._active_learn_algorithm.make_iteration(indexes, answers)
