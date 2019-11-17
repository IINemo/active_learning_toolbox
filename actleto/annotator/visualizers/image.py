import numpy as np
from ipywidgets import Image as WidgImage
from PIL import Image
import io


class ImageVisualizer(object):
    """Visualizer for images.
    
    This visualizer can display images (e.g., MNIST), which are stored as rows 
    in a dataframe.
    """
    
    def __init__(self, columns_range, img_shape, img_mode, preview_shape):
        """ImageVisualizer constructor. 
        
        Args:
            rng (tuple): tuple (start, end) - range of columns in pandas.DataFrame, which contain image data
            img_shape (tuple): original image shape width x height.
            img_format (str): image format: "L" - black&white (MNIST); "RGB"; "CMYK"; "1".
            preview_shape (tuple): output image size.
            
        """
        super().__init__()
        self._columns_range = columns_range
        self._img_shape = img_shape
        self._img_mode = img_mode
        self._preview_shape = preview_shape
    
    def init(self, dataframe, answers):
        self._dataframe = dataframe
        self._answers = answers
    
    def __call__(self, index):
        """Invokes the visualizer.
        
        Args:
            dataframe (pandas.DataFrame): the dataframe that contains the data for visualization.
            index (int): the positional (iloc) index of the row to visualize.
            
        Returns:
            tuple: The list of widgets that visualize the row with number index.
            
        """
        img_array = self._dataframe.iloc[index][self._columns_range[0] : 
                                                self._columns_range[1]].as_matrix()

        if img_array.shape[0] > np.product(self._img_shape):
            cur_img_shape = self._img_shape + (-1,)
        else:
            cur_img_shape = self._img_shape

        img = Image.fromarray(img_array.reshape(cur_img_shape), self._img_mode)
        
        buffer = io.BytesIO()
        img.convert('RGB').save(buffer, format = 'PNG')
        
        return (WidgImage(value = buffer.getvalue(), 
                          format = 'PNG', 
                          width = self._preview_shape[0], 
                          height = self._preview_shape[1]),)
