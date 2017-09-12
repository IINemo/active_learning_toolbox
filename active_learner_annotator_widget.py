from traitlets import Int, Dict, Instance, observe
from ipywidgets import Button, VBox, HBox, HTML, Box, Layout, Label, Textarea, Text, ToggleButtons
import pandas as pd
import numpy as np
from .visualizer_textarea import VisualizerTextArea


class ActiveLearnerAnnotatorWidget(Box):
    """The widget for Jupyter that implements example annotator.
    
    The widget can be used without active learning.
    
    """
    
    _max_examples = Int(5, allow_none = False).tag(sync=True)
    _current_position = Int(0, allow_none = False).tag(sync = True)
    _dataframe = Instance(klass = pd.DataFrame)
    
    _example_layout = Instance(klass=Layout)
    _example_layout = Layout(width = '100%', 
                             border = 'solid 2px', 
                             margin = '3px', 
                             align_items = 'stretch',
                             padding = '2px')
    
    ANNOT_DONT_KNOW = "Don't know"

    def __init__(self, 
                 dataframe, 
                 answers = None,
                 max_examples = 5,
                 current_position = 0, 
                 textual_labels = [], 
                 drop_labels = [],
                 y_labels = {'True' : True,
                             'False' : False},
                 visualizer = None,
                 display_feature_table = True,
                 *args, **kwargs):
        """Annotator constructor.
        
        Args:
            dataframe (pandas.DataFrame):  the dataframe that contains data for examples visualizatin.
            answers (numpy.array): the array that contains known answers 
                (that will not be marked as Don't Know check button).
            max_examples (int): maximum number of examples per page.
            current_position (int): index of position, from which iteration should start.
            textual_labels (list): list of string labels that will be visualized with VisualizerTextArea.
            drop_labels (list): list of string labels that will be dropped from visualization via table.
            y_labels (dict): dict {<y_textual_label> : <y_value>}.
            visualizer (object): visualizer for X_helper representation. The default is None. If None the widget
                will invoke VisualizerTextArea by deafult.
            
        """
        super(Box, self).__init__(*args, **kwargs)
        
        self._y_labels = y_labels
        self._y_labels[self.ANNOT_DONT_KNOW] = None
        self._y_labels_reversed = {v : k for k, v in self._y_labels.items()}
        self._display_feature_table = display_feature_table
        
        self._dataframe = dataframe
        self._current_position = min(current_position, self._dataframe.shape[0] - 1)
        self._max_examples = max_examples
        self._answers = (answers if answers is not None 
                         else np.array([None] * self._dataframe.shape[0]))
        assert self._answers.shape[0] == self._dataframe.shape[0], \
               'The length of dataframe should match the length of numpy.array with answers.'
        
        self._drop_labels = drop_labels
        self._visualizer = visualizer
        if self._visualizer is None:
            self._visualizer = VisualizerTextArea(textual_labels)
            self._drop_labels = textual_labels
        
        self._draw()
        self.observe(self._draw, names='_current_position')
        
    def get_answers(self):
        """Returns numpy.array with answers."""
        return self._answers
    
    def get_dataframe(self):
        """Returns pandas.DataFrame with feature values."""
        return self._dataframe
        
    def _click_prev(self, button):
        self._current_position = max(self._current_position - self._max_examples, 0)
            
    def _click_next(self, button):
        next_position = self._current_position + self._max_examples
        if next_position < self._dataframe.shape[0]:
            self._current_position = next_position
            
    def _int_text_value_changed(self, wdg):
        try:
            new_value = int(wdg.value)
        except ValueError:
            return
        
        if new_value < 0:
            new_value = self._dataframe.shape[0] + new_value
            if new_value < 0:
                return
            
        if new_value >= self._dataframe.shape[0]:
            return
            
        self._current_position = new_value
            
    def _make_controls(self):
        controls = HBox(children = [Button(description='Prev'), 
                                    Button(description='Next'),
                                    Text(value = str(self._current_position), 
                                         layout = Layout(width = '80px')),
                                    Label(value = 'out of', 
                                          layout = Layout(width = '35px')),
                                    Text(value = str(self._dataframe.shape[0]), 
                                         disabled = True,
                                         layout = Layout(width = '80px'))])
        
        controls.children[0].on_click(self._click_prev)
        controls.children[1].on_click(self._click_next)
        controls.children[2].on_submit(self._int_text_value_changed)

        return controls
    
    def _annotate(self, num, change):
        ch = change['new']
        if ch == self.ANNOT_DONT_KNOW:
            self._answers[num] = None
        else:
            self._answers[num] = self._y_labels[ch]
    
    def _answer_to_label(self, answer):
        return self._y_labels_reversed[answer]
    
    def _draw(self, change = None):
        self._table = VBox(layout = Layout(width = '100%'))
        self._table.children += (self._make_controls(),)
        
        last_element = self._current_position + self._max_examples
        if last_element > self._dataframe.shape[0]:
            last_element = self._dataframe.shape[0]
        
        for i in range(self._current_position, last_element):
            data_row = VBox(layout = self._example_layout)
            
            if self._display_feature_table:
                elem = self._dataframe.iloc[i, :].drop(self._drop_labels)
                data_row.children += (HTML(value = pd.DataFrame([elem.values], 
                                                                columns = elem.index, 
                                                                index = [self._dataframe.index[i]])
                                           .to_html(classes=['table', 'table-striped'])),)

            data_row.children += self._visualizer(self._dataframe, i)
            
            data_row.children += (ToggleButtons(options=([self.ANNOT_DONT_KNOW] + 
                                                [k for k in self._y_labels.keys() if k != self.ANNOT_DONT_KNOW]),
                                                value = self._answer_to_label(self._answers[i]),
                                                description='Your annotation:',
                                                disabled=False),)
            data_row.children[-1].observe(lambda tgl_bt, num = i: self._annotate(num, tgl_bt), 
                                          names='value')

            self._table.children += (data_row,)
        
        self._table.children += (self._make_controls(),)
        self.children = (self._table,)
    