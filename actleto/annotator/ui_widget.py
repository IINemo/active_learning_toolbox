from .annotator_widget import AnnotatorWidget

from ipywidgets import Button, VBox, HBox, Label
import pandas as pd
import numpy as np
import logging
from threading import Timer
import os


logger = logging.getLogger('actleto')


def prep_log(obj):
    return '\n' + str(obj)


class EvaluationCallbackLogging(object):
    def __init__(self, logger):
        self._logger = logger
        
    def __call__(self, eval_res):
        self._logger.info('Evaluation: {}'.format(prep_log(pd.DataFrame([eval_res])
                                                           .to_string(index=False))))

    
class ActiveLearnerUiWidget(VBox):
    """The main ui widget for active learning annotation.
    
    Create widget in Jupyter, configure it with ActiveLearner object and invoke.
    
    """
    
    _reset_check_time = 4
    
    def __init__(self, 
                 active_learner, 
                 X_helper,
                 visualize_columns = [],
                 drop_labels = [],
                 display_feature_table = True,
                 y_labels = {"True" : True, 
                             "False" : False},
                 visualizer = None,
                 save_path = 'annotation',
                 evaluation_callback = None,
                 save_time = 0,
                 *args, **kwargs):
        """Widget constructor.
        
        Args:
            active_learner (ActiveLearner): the ActiveLearner object configured with query strategy.
            X_helper (pandas.DataFrame): the dataframe with data for visualization.
            textual_labels (list): list of string labels that will be visualized with VisualizerTextArea.
            drop_labels (list): list of string labels that will be dropped from visualization via table.
            y_labels (dict): dict {<y_textual_label> : <y_value>}.
            y_visualizer (object): visualizer for X_helper representation. The default is None. If None the widget
                will invoke VisualizerTextArea by deafult.
            save_path (str): the path to save the results.
            evaluation_callback (functor): the callback for evaluation. The default is logging callback.
            save_time (int): Autosave time. If 0 then autosave is disabled. If u use auto save u have to 
                call stop() method to disabel autosave in the current widget.
            
        """
        super(VBox, self).__init__(*args, **kwargs)

        self._X_helper = X_helper
        self._active_learner = active_learner
        self._save_path = save_path
        self._evaluation_callback = evaluation_callback or EvaluationCallbackLogging(logger)

        self._y_labels = y_labels
        self._visualizer = visualizer
        self._drop_labels = drop_labels
        self._visualize_columns = visualize_columns
        self._display_feature_table = display_feature_table
        
        controls = HBox()
        
        self._button_next_iter = Button(description = 'Next iteration')
        self._button_next_iter.on_click(self._click_next_iteration)
        controls.children += (self._button_next_iter,)
        
        self._iteration_num = 0
        controls.children += (Label(self._iteration_label()),)
        
        self._button_save = Button(description = 'Save')
        self._button_save.on_click(self._click_save)
        controls.children += (self._button_save,)

        self.children = (controls, self._make_annotator_widget())
        
        self._save_time = save_time
        self._timer = None
        if self._save_time > 0:
            self._start_save_timer()
            
        self._timer_check_save_reset = None
        self._timer_check_next_iteration_reset = None
    
    def __del__(self):
        self.stop()
        
    def get_active_learner(self):
        """Returns the active learner object that was delivered to the constructor."""
        return self._active_learner
    
    def stop(self):
        if self._timer is not None:
            self._timer.cancel()
    
    def _save_on_timer(self):
        logger.info('Autosave.')
        self._save_answers(os.path.splitext(self._save_path)[0] + '_autosave')
        self._start_save_timer()
        
    def _start_save_timer(self):
        self._timer = Timer(self._save_time, self._save_on_timer)
        self._timer.start()
        
    def _get_annotator_widget(self):
        return self.children[1]
    
    def _iteration_label(self):
        return 'Iteration #{}'.format(self._iteration_num)
    
    def _increment_iteration_num(self):
        self._iteration_num += 1
        self.children[0].children[1].value = self._iteration_label()
        
    def _make_annotator_widget(self):
        samples_to_annotate = self._active_learner.choose_samples_for_annotation()
        return AnnotatorWidget(dataframe = self._X_helper.iloc[samples_to_annotate],
                               visualize_columns = self._visualize_columns,
                               drop_labels = self._drop_labels,
                               visualizer = self._visualizer,
                               display_feature_table = self._display_feature_table,
                               y_labels = self._y_labels)
    
    def _click_next_iteration(self, button):
        if self._timer_check_next_iteration_reset:
            self._timer_check_next_iteration_reset.cancel()
        
        self._button_next_iter.disabled = True
        self._button_next_iter.icon = 'clock-o'
        annotated_indexes = [self._X_helper.index.get_loc(e) 
                             for e in self._get_annotator_widget().get_dataframe().index]
        
        self._active_learner.make_iteration(annotated_indexes, 
                                            self._get_annotator_widget().get_answers())
        
        logger.info(self._iteration_label())
        eval_res = self._active_learner.evaluate()
        if eval_res is not None:
            self._evaluation_callback(eval_res)
        
        self._increment_iteration_num()
        self.children = (self.children[0], self._make_annotator_widget())
        self._button_next_iter.icon = 'check'
        self._button_next_iter.disabled = False
        
        self._timer_check_next_iteration_reset = Timer(self._reset_check_time, 
                                                       self._check_next_iteration_reset)
        self._timer_check_next_iteration_reset.start()
        
    def _check_next_iteration_reset(self):
        self._button_next_iter.icon = ''
                             
    def _click_save(self, button):
        if self._timer_check_save_reset:
            self._timer_check_save_reset.cancel()
        
        self._button_save.disabled = True
        self._button_save.icon = 'clock-o'
        self._save_answers(self._save_path)
        self._button_save.icon = 'check'
        self._button_save.disabled = False
        
        self._timer_check_save_reset = Timer(self._reset_check_time, self._check_save_reset)
        self._timer_check_save_reset.start()
                                             
    def _check_save_reset(self):
        self._button_save.icon = ''
                                             
    def _save_answers(self, path):
        np.save(path, self._active_learner.get_annotation())
        logger.info('Saved. File path: {}'.format(os.path.splitext(path)[0] + '.npy'))
        