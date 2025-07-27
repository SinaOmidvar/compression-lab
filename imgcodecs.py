import cv2
import pywt
import numpy as np
import utils
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
    normalized_root_mse as nrmse,
    normalized_mutual_information as nmi,
)

# Image transformation
def dct(ycc_img, lum_ratio='4:4:4', chr_ratio='4:2:0',
                 block_h=8, block_w=8, quality=50):
    
    ##############################################
    # Instantiation
    ##############################################
    
    lum_downsample = utils.Downsampling(ratio=lum_ratio)
    chr_downsample = utils.Downsampling(ratio=chr_ratio)
    image_block = utils.ImageBlock(block_height=block_h, block_width=block_w)
    padding = utils.Padding(mode="reflect", block_height=block_h, block_width=block_w)
    dct2d = utils.DCT2D(norm='ortho')
    quantization = utils.Quantization(quality=quality)
    
    ##############################################
    # Preprocess
    ##############################################
    
    # Center
    ycc_img = ycc_img.astype(int) - 128

    # Padding if necessary
    ycc_img = padding.add(ycc_img)

    # Downsampling
    Y, Cr, Cb = cv2.split(ycc_img)
    Y = lum_downsample(Y)
    Cr = chr_downsample(Cr)
    Cb = chr_downsample(Cb)
    ycc_img = np.stack((Y, Cr, Cb), axis=2)

    # Create 8x8 blocks
    blocks, indices = image_block.forward(ycc_img)
    
    ##############################################
    # Compression (Sequential Processing)
    ##############################################
    
    compressed = []
    for block, index in zip(blocks, indices):
        # DCT
        encoded = dct2d.forward(block)
        if index[2] == 0:
            channel_type = 'lum'
        else:
            channel_type = 'chr'
            
        # Quantization
        encoded = quantization.forward(encoded, channel_type)
        
        # Dequantization
        decoded = quantization.backward(encoded, channel_type)
        
        # Reverse DCT
        compressed_block = dct2d.backward(decoded)
        compressed.append(compressed_block)

    compressed = np.array(compressed)
    
    ##############################################
    # Postprocess
    ##############################################
    
    # Reconstruct image from blocks
    ycc_img_compressed = image_block.backward(compressed, indices)

    # Remove padding
    ycc_img_compressed = padding.remove(ycc_img_compressed)

    # Rescale
    ycc_img_compressed = (ycc_img_compressed + 128).astype('uint8')
    
    # Return YCrCb
    return ycc_img_compressed

def dwt(ycc_img, lum_ratio='4:4:4', chr_ratio='4:2:0', 
                 method='threshold', wavelet='db8', level=4,
                 threshold=10.0, threshold_mode='soft', quant_step=20.0):
    
    ##############################################
    # Instantiation
    ##############################################
    
    lum_downsample = utils.Downsampling(ratio=lum_ratio)
    chr_downsample = utils.Downsampling(ratio=chr_ratio)
    padding = utils.Padding(mode="reflect", block_height=2, block_width=2)
    
    ##############################################
    # Preprocess
    ##############################################
    
    # Padding if necessary
    ycc_img = padding.add(ycc_img)
    
    # Convert to float32 for processing
    ycc_img = ycc_img.astype(np.float32)

    # Downsampling
    Y, Cr, Cb = cv2.split(ycc_img)
    Y = lum_downsample(Y)
    Cr = chr_downsample(Cr)
    Cb = chr_downsample(Cb)

    ##############################################
    # Compression 
    ##############################################
    
    def process_channel(channel):
        """Apply wavelet compression to a single channel"""
        # Decompose
        coeffs = pywt.wavedec2(channel, wavelet, level=level)
        coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        
        # Process coefficients
        if method == 'threshold':
            coeff_arr = pywt.threshold(coeff_arr, threshold, mode=threshold_mode, substitute=0)
        elif method == 'quantize':
            coeff_arr = np.round(coeff_arr / quant_step) * quant_step
        
        # Reconstruct
        coeffs = pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format='wavedec2')
        rec = pywt.waverec2(coeffs, wavelet)
        
        # Handle size mismatches
        h, w = channel.shape
        rec = rec[:h, :w]
        return np.clip(rec, 0, 255)
    
    # Process all channels
    Y = process_channel(Y)
    Cr = process_channel(Cr)
    Cb = process_channel(Cb)
    
    ##############################################
    # Postprocess
    ##############################################
    
    # Reconstruct image from channels 
    ycc_img_compressed = np.stack((Y, Cr, Cb), axis=2)
    
    # Remove padding
    ycc_img_compressed = padding.remove(ycc_img_compressed)

    # Rescale
    ycc_img_compressed = np.clip(ycc_img_compressed, 0, 255).astype('uint8')
    
    # Return YCrCb
    return ycc_img_compressed

# Utility functions
def rgb2ycc(rgb_img):
    return cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2YCrCb)

def ycc2rgb(ycc_img):
    return cv2.cvtColor(np.array(ycc_img), cv2.COLOR_YCrCb2RGB)

# Quality metrics
def metrics(original, compressed):
    # Convert to Gray
    original = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2GRAY)
    compressed = cv2.cvtColor(np.array(compressed), cv2.COLOR_RGB2GRAY)
    # Compute metrics
    metrics = {
        "PSNR": psnr(original, compressed, data_range=255),
        "SSIM": ssim(original, compressed, data_range=255),
        "NRMSE": nrmse(original, compressed, normalization='mean'),
        "NMI": nmi(original, compressed)
    }
    return {k: round(v, 2) for k, v in metrics.items()}



    