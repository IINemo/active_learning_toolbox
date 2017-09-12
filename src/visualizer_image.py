from ipywidgets import Layout, Label
from ipywidgets import Image as WidgImage
from PIL import Image
import io


class VisualizerImage(object):
    """Visualizer for images.
    
    This visualizer can display images, presented as one dimentional 
    numpy.arrays (e.g., MNIST).
    
    """
    
    def __init__(self, rng, img_shape, img_format, preview_shape):
        """VisualizerImage constructor. 
        
        Args:
            rng (tuple): tuple (start, end) - range of image in the pandas array.
            img_shape (tuple): original image shape width x height.
            img_format (str): image format: "L" - black&white (MNIST); "RGB"; "CMYK"; "1".
            preview_shape (tuple): output image size.
            
        """
        super().__init__()
        self._rng = rng
        self._img_shape = img_shape
        self._img_format = img_format
        self._preview_shape = preview_shape
    
    def __call__(self, dataframe, index):
        """Invokes the visuzlizer.
        
        Args:
            dataframe (pandas.DataFrame): the dataframe that contains the data for visualization.
            index (int): the positional (iloc) index of the row to visualize.
            
        Returns:
            tuple: The tuple of widgets that visualize the row with number index.
            
        """
        result = ()
        img_array = dataframe.iloc[index][self._rng[0] : self._rng[1]].as_matrix()
        img = Image.fromarray(img_array.reshape(self._img_shape), self._img_format)
        
        buffer = io.BytesIO()
        img.convert('RGB').save(buffer, format = 'PNG')
        
        result += (WidgImage(value = buffer.getvalue(), 
                             format = 'PNG', 
                             width = self._preview_shape[0], 
                             height = self._preview_shape[1]),)
        return result
