import numpy as np
from scipy.fft import dct
from scipy.signal import convolve2d

class Downsampling():
    def __init__(self, ratio='4:2:0'):
        assert ratio in ('4:4:4', '4:2:2', '4:2:0'), "Please choose one of the following {'4:4:4', '4:2:2', '4:2:0'}"
        self.ratio = ratio
        
    def __call__(self, x):
        # No subsampling
        if self.ratio == '4:4:4':
            return x
        else:
            # Downsample with a window of 2 in the horizontal direction
            if self.ratio == '4:2:2':
                kernel = np.array([[0.5], [0.5]])
                out = np.repeat(convolve2d(x, kernel, mode='valid')[::2,:], 2, axis=0)
            # Downsample with a window of 2 in both directions
            else:
                kernel = np.array([[0.25, 0.25], [0.25, 0.25]])
                out = np.repeat(np.repeat(convolve2d(x, kernel, mode='valid')[::2,::2], 2, axis=0), 2, axis=1)
            return np.round(out).astype('int')

class Padding():
    def __init__(self, mode='reflect', block_height=8, block_width=8):
        self.mode = mode
        self.block_height = block_height
        self.block_width = block_width
        self.left_padding = self.right_padding = self.top_padding = self.bottom_padding = 0
        
    def add(self, image):
        
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        self.image_channel = image.shape[2]
    
        # Vertical padding
        if self.image_height % self.block_height != 0:
            vpad = self.block_height - (self.image_height % self.block_height)
            self.top_padding = vpad // 2 
            self.bottom_padding = vpad - self.top_padding
            image = np.concatenate((np.repeat(image[:1], self.top_padding, 0), image, 
                                    np.repeat(image[-1:], self.bottom_padding, 0)), axis=0)
            
        # Horizontal padding
        if self.image_width % self.block_width != 0:
            hpad = self.block_width - (self.image_width % self.block_width)
            self.left_padding = hpad // 2 
            self.right_padding = hpad - self.left_padding
            image = np.concatenate((np.repeat(image[:,:1], self.left_padding, 1), image, 
                                    np.repeat(image[:,-1:], self.right_padding, 1)), axis=1)
    
        # Update dimension
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        
        return image
    
    def remove(self, image):
        # Remove padding
        if self.top_padding > 0:
            image = image[self.top_padding:,:,:]
        if self.bottom_padding > 0:
            image = image[:-self.bottom_padding,:,:] 
        if self.left_padding > 0:
            image = image[:,self.left_padding:,:]
        if self.right_padding > 0:
            image = image[:,:-self.right_padding,:]
        return image

class ImageBlock():
    def __init__(self, block_height=8, block_width=8):
        self.block_height = block_height
        self.block_width = block_width
    
    def forward(self, image):
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        self.image_channel = image.shape[2]
    
        # Create blocks
        blocks = []
        indices = []
        for i in range(0, self.image_height, self.block_height):
            for j in range(0, self.image_width, self.block_width):
                for k in range(self.image_channel):
                    blocks.append(image[i:i+self.block_height, j:j+self.block_width, k])
                    indices.append((i,j,k))
                    
        blocks = np.array(blocks)
        indices = np.array(indices)
        return blocks, indices
    
    def backward(self, blocks, indices):
        # Empty image array
        image = np.zeros((self.image_height, self.image_width, self.image_channel)).astype(int)
        for block, index in zip(blocks, indices):
            i, j, k = index
            image[i:i+self.block_height, j:j+self.block_width, k] = block
            
        return image

class DCT2D():
    def __init__(self, norm='ortho'):
        if norm is not None:
            assert norm == 'ortho', "norm needs to be in {None, 'ortho'}"
        self.norm = norm
    
    def forward(self, x):
        out = dct(dct(x, norm=self.norm, axis=0), norm=self.norm, axis=1)
        return out
    
    def backward(self,x):
        out = dct(dct(x, type=3, norm=self.norm, axis=0), type=3, norm=self.norm, axis=1)
        return np.round(out)

class Quantization():
    # Standard quantization matrices
    Q_lum = np.array([[16,11,10,16,24,40,51,61],
                      [12,12,14,19,26,58,60,55],
                      [14,13,16,24,40,57,69,56],
                      [14,17,22,29,51,87,80,62],
                      [18,22,37,56,68,109,103,77],
                      [24,35,55,64,81,104,113,92],
                      [49,64,78,87,103,121,120,101],
                      [72,92,95,98,112,100,103,99]])
    
    Q_chr = np.array([[17,18,24,47,99,99,99,99],
                      [18,21,26,66,99,99,99,99],
                      [24,26,56,99,99,99,99,99],
                      [47,66,99,99,99,99,99,99],
                      [99,99,99,99,99,99,99,99],
                      [99,99,99,99,99,99,99,99],
                      [99,99,99,99,99,99,99,99],
                      [99,99,99,99,99,99,99,99]])
    
    def __init__(self, quality=50):
        """
        Initialize with quality factor (0-100 scale)
        Default quality is 50 (standard JPEG matrices)
        """
        self.quality = self._clamp_quality(quality)
        self.scale_factor = self._calculate_scale_factor()
        self.scaled_Q_lum = self.get_quantization_matrix('lum')
        self.scaled_Q_chr = self.get_quantization_matrix('chr')
    
    def _clamp_quality(self, quality):
        """Ensure quality stays within 1-100 range"""
        return max(1, min(100, quality))
    
    def _calculate_scale_factor(self):
        """
        Calculate scaling factor based on quality
        - Low quality (0-50): More aggressive quantization
        - High quality (50-100): Less quantization
        """
        if self.quality < 50:
            return (50 / self.quality)**2
        else:
            return (100 - self.quality) / 50
    
    def get_quantization_matrix(self, channel_type):
        """
        Return scaled quantization matrix based on channel type and quality
        """
        assert channel_type in ('lum', 'chr')
        
        # Select base matrix
        base_matrix = self.Q_lum if channel_type == 'lum' else self.Q_chr
        
        # Scale matrix according to quality
        scaled_matrix = base_matrix * self.scale_factor
        
        # Ensure values stay within valid range (1-550)
        scaled_matrix = np.clip(scaled_matrix, 1, 550)
        
        # For low quality: Round to integers for more aggressive quantization
        # For high quality: Keep fractional values for finer gradations
        if self.quality < 50:
            return np.round(scaled_matrix)
        return scaled_matrix
    
    def forward(self, x, channel_type):
        Q = self.scaled_Q_lum if channel_type == 'lum' else self.scaled_Q_chr
        return np.round(x / Q)
    
    def backward(self, x, channel_type):
        Q = self.scaled_Q_lum if channel_type == 'lum' else self.scaled_Q_chr
        return x * Q