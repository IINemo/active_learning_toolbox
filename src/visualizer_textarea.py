from ipywidgets import Layout, Label, Textarea, Text
from traitlets import Instance


class VisualizerTextArea(object):
    """Visualizer for texts via text areas."""
    
    def __init__(self, textual_labels, width = '90%', height = '150px'):
        """
        Args:
            textual_labels (list): List of labels (index elements) of columns that should be
                visualized via textarea.
            width (str): width of the text area.
            height (str): height of the text are.
        
        """
        super().__init__()
        self._textual_labels = textual_labels
        self._text_layout = Instance(klass=Layout)
        self._text_layout = Layout(width = width, height = height)
    
    def __call__(self, dataframe, index):
        """Invokes the visuzlizer.
        
        Args:
            dataframe (pandas.DataFrame): the dataframe that contains the data for visualization.
            index (int): the positional (iloc) index of the row to visualize.
            
        Returns:
            tuple: The tuple of widgets that visualize the row with number index.
            
        """
        result = ()
        for label in self._textual_labels:
            result += (Label('{}:'.format(label)),)
            result += (Textarea(value = str(dataframe.iloc[index][label]), 
                                layout = self._text_layout,
                                disabled = True),)
            
        return result
