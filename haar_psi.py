import numpy
import matplotlib.pyplot as plt
from scipy import signal

def haar_psi(reference_image, distorted_image, subsampling = True):
    
    # Converts the image values to double precision floating point numbers
    reference = reference_image.astype(numpy.float64)
    distorted = distorted_image.astype(numpy.float64)
    
    # The HaarPSI algorithm requires two constants, C and alpha, that have been experimentally determined
    # to be C = 30 and alpha = 4.2
    C = 30.0
    alpha = 4.2
    
    # Subsamples the images, which simulates the typical distance between an image and its viewer
    if subsampling:
        reference = subsample(reference)
        distorted = subsample(distorted)
        
    # Performs the Haar wavelet decomposition
    number_of_scales = 3
    coefficients_reference = haar_wavelet_decompose(reference, number_of_scales)
    coefficients_distorted = haar_wavelet_decompose(distorted, number_of_scales)
    
    # Pre-allocates the variables for the local similarities and the weights
    local_similarities = numpy.zeros(sum([reference.shape, (2, )], ()))
    weights = numpy.zeros(sum([reference.shape, (2, )], ()))
    
    # Computes the weights and similarities for each orientation
    for orientation in range(2):
        weights[:, :, orientation] = numpy.maximum(
            numpy.abs(coefficients_reference[:, :, 2 + orientation * number_of_scales]),
            numpy.abs(coefficients_distorted[:, :, 2 + orientation * number_of_scales])
        )
        coefficients_reference_magnitude = numpy.abs(coefficients_reference[:, :, (orientation * number_of_scales, 1 + orientation * number_of_scales)])
        coefficients_distorted_magnitude = numpy.abs(coefficients_distorted[:, :, (orientation * number_of_scales, 1 + orientation * number_of_scales)])
        local_similarities[:, :, orientation] = numpy.sum(
            (2 * coefficients_reference_magnitude * coefficients_distorted_magnitude + C) / (coefficients_reference_magnitude**2 + coefficients_distorted_magnitude**2 + C),
            axis = 2
        ) / 2
    
    # Calculates the final score
    similarity = logit(numpy.sum(sigmoid(local_similarities[:], alpha) * weights[:]) / numpy.sum(weights[:]), alpha)**2

    # Returns the result
    return round(similarity, 2), local_similarities, weights
    
def subsample(image):
    subsampled_image = signal.convolve2d(image, numpy.ones((2, 2)) / 4.0, mode = "same")
    subsampled_image = subsampled_image[::2, ::2]
    return subsampled_image

def haar_wavelet_decompose(image, number_of_scales):
    
    coefficients = numpy.zeros(sum([image.shape, (2 * number_of_scales, )], ()))
    for scale in range(1, number_of_scales + 1):
        haar_filter = 2**(-scale) * numpy.ones((2**scale, 2**scale))
        haar_filter[:haar_filter.shape[0] // 2, :] = -haar_filter[:haar_filter.shape[0] // 2, :]
        coefficients[:, :, scale - 1] = signal.convolve2d(image, haar_filter, mode = "same")
        coefficients[:, :, scale + number_of_scales - 1] = signal.convolve2d(image, numpy.transpose(haar_filter), mode = "same")
        
    return coefficients    

def sigmoid(value, alpha):
    return 1.0 / (1.0 + numpy.exp(-alpha * value))

def logit(value, alpha):
    return numpy.log(value / (1 - value)) / alpha

def visualize_haarpsi_maps(local_similarities, weights):
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    
    # Similarity Maps
    # Adjust if identical images
    vmin = numpy.min(local_similarities) if numpy.min(local_similarities) < 1 else 0  
    axes[0, 0].imshow(local_similarities[:, :, 0], cmap='Blues', vmin=vmin, vmax=1)
    axes[0, 1].imshow(local_similarities[:, :, 1], cmap='Blues', vmin=vmin, vmax=1)
    
    # Weight Maps (normalized for visibility)
    axes[1, 0].imshow(weights[:, :, 0], cmap='hot', vmin=0, vmax=numpy.percentile(weights, 99))
    axes[1, 1].imshow(weights[:, :, 1], cmap='hot', vmin=0, vmax=numpy.percentile(weights, 99))
    
    titles = ['Horizontal Similarity', 'Vertical Similarity', 'Horizontal Weights', 'Vertical Weights']
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    return fig
