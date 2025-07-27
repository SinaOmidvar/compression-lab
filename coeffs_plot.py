import numpy as np
import pywt
import matplotlib.pyplot as plt

def depict(ycc_img, chan='Y', mode='Square', wavelet='db1', level=2, cmap='bone', gamma=0.3):

    YCrCb = {"Y": 0, "Cr": 1, "Cb": 2}
    index = YCrCb[chan]
    channel = ycc_img[:,:,index]
    
    # Convert to float and normalize
    channel = channel.astype(float)/255.0

    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(channel, wavelet, mode='periodization', level=level)
    
    cDetail = coeffs[1:]
    cDetail.reverse()
    coeffs[1:] = cDetail
    
    # Enhanced visualization function
    def enhance_coeffs(coeffs):
        abs_coeffs = np.abs(coeffs)
        norm = (abs_coeffs - np.min(abs_coeffs)) / (np.max(abs_coeffs) - np.min(abs_coeffs) + 1e-8)
        enhanced = norm ** gamma
        cmap_obj = plt.get_cmap(cmap)
        return cmap_obj(enhanced)
    
    # Hierarchical visualization
    def hierarchical():
        # Create figure object explicitly
        rows = level + 1
        fig = plt.figure(figsize=(12, 3*rows))  
         
        # Plot channel
        ax1 = fig.add_subplot(rows, 4, 1)
        ax1.imshow(channel, cmap='gray')
        ax1.set_title(f"{chan} Channel")
        ax1.axis('off')
        
        titles = ['Horizontal', 'Vertical', 'Diagonal']
        
        cA = coeffs[0]  # Approximation coefficients
        cDetail = coeffs[1:]  # Detail coefficients
        
        # Plot decomposition for each level
        for lvl in range(level):
            start_idx = 4*(lvl+1) + 1
            
            # Plot approximation (only for last level)
            if lvl == level-1:
                ax = fig.add_subplot(rows, 4, start_idx)
                if cmap in ["Blues", "Greys"]:
                    ax.imshow(1 - cA, cmap=cmap)
                else:
                    ax.imshow(cA, cmap=cmap)
                ax.set_title(f'Approximation Level {lvl+1}')
                ax.axis('off')
                start_idx += 1
            
            # Plot details
            for i in range(3):
                ax = fig.add_subplot(rows, 4, start_idx + i)
                ax.imshow(enhance_coeffs(cDetail[lvl][i]))
                ax.set_title(f'{titles[i]} Detail Level {lvl+1}')
                ax.axis('off')
        
        plt.tight_layout()
        return fig

    # Square visualization
    def square():
        # Create the main figure
        grid_size = 2 ** level
        fig_size = 6 + grid_size / 2
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(fig_size, fig_size))
        
        # Hide all axes initially
        for ax_row in axes:
            for ax in ax_row:
                ax.axis('off')
        
        # Function to plot coefficients recursively
        def plot_level(ax_top_left, size, coeffs, current_level):
            # Calculate quadrant size (half of current size)
            q_size = size // 2
            
            # Base case: current_level = 0 (approximation coefficients)
            if current_level == 0:
                # Plot approximation coefficients
                cA = coeffs[0]
                if cmap == "Blues":
                    ax_top_left.imshow(1 - cA, cmap=cmap)
                else:
                    ax_top_left.imshow(cA, cmap=cmap)
                ax_top_left.axis('off')
                return
            
            # Get coefficients for this level
            cA, cH, cV, cD = coeffs[0], coeffs[1][0], coeffs[1][1], coeffs[1][2]
            next_coeffs = coeffs[0] if len(coeffs) > 2 else None
            
            # Calculate positions for the 4 quadrants
            top, left = ax_top_left.get_subplotspec().rowspan.start, ax_top_left.get_subplotspec().colspan.start
            
            # Top-left quadrant: Approximation coefficients (recurse)
            a_ax = axes[top, left]
            if next_coeffs is not None:
                plot_level(a_ax, q_size, [cA] + coeffs[2:], current_level-1)
            else:
                if cmap == "Blues":
                    a_ax.imshow(1 - cA, cmap=cmap)
                else:
                    a_ax.imshow(cA, cmap=cmap)
                a_ax.axis('off')
            
            # Top-right quadrant: Horizontal details
            h_ax = plt.subplot2grid((grid_size, grid_size), (top, left + q_size), rowspan=q_size, colspan=q_size)
            h_ax.imshow(enhance_coeffs(cH))
            # h_ax.annotate('Horizontal', (5, 10))
            h_ax.axis('off')
            
            # Bottom-left quadrant: Vertical details
            v_ax = plt.subplot2grid((grid_size, grid_size), (top + q_size, left), rowspan=q_size, colspan=q_size)
            v_ax.imshow(enhance_coeffs(cV))
            v_ax.axis('off')
            
            # Bottom-right quadrant: Diagonal details
            d_ax = plt.subplot2grid((grid_size, grid_size), (top + q_size, left + q_size), rowspan=q_size, colspan=q_size)
            d_ax.imshow(enhance_coeffs(cD))
            d_ax.axis('off')
        
        # Start recursive plotting from the top-left corner
        plot_level(axes[0, 0], grid_size, coeffs, level)
        
        plt.tight_layout()
        return fig

    if mode == 'Square':
        return square()
    elif mode == 'Hierarchical':
        return hierarchical()
    
    

