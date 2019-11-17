from ipywidgets import Layout, Label, Textarea


class TextAreaVisualizer(object):
    """Visualizer for texts via text areas."""

    def __init__(self, text_columns, width = '90%', height = '150px'):
        """
        Args:
            text_columns (list): List of labels (index elements) of columns that should be
                visualized via textarea.
            width (str): width of the text area.
            height (str): height of the text are.
        """
        super().__init__()
        self._text_columns = text_columns
        self._text_layout = Layout(width = width, height = height)
        
    def init(self, dataframe, answers):
        self._dataframe = dataframe
        self._answers = answers

    def __call__(self, index):
        """Invokes the visuzlizer.

        Args:
            dataframe (pandas.DataFrame): the dataframe that contains the data for visualization.
            index (int): the positional (iloc) index of the row to visualize.

        Returns:
            list: widgets that visualize the row
        """
        result = []
        row = self._dataframe.iloc[index]
        for label in self._text_columns:
            result.append(Label('{}:'.format(label)))
            result.append(Textarea(value = str(row[label]),
                                   layout = self._text_layout,
                                   disabled = True))
            
        return tuple(result)
