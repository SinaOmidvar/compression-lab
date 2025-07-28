import streamlit as st
from streamlit_image_comparison import image_comparison
from PIL import Image
import imgcodecs
import coeffs_plot
import os
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Image Compression Lab",
    layout="wide",
    page_icon="🧪",
    initial_sidebar_state="expanded"
)

# Import CSS file
css_path = os.path.join("assets", "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Session state initialization
def init_session():
    if 'original_img' not in st.session_state:
        st.session_state.original_img = None
    if 'compressed_img' not in st.session_state:
        st.session_state.compressed_img = None
    if 'compression_history' not in st.session_state:
        st.session_state.compression_history = []

init_session()

# Image Upload and Preview
def upload_preview():
    # Image upload
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "tiff", "WebP", "bmp"])
    if uploaded_file:
        img = Image.open(uploaded_file)

        # Conversion
        if img.mode == "L":
            st.session_state.original_img = img.convert("RGB")
        else:
            st.session_state.original_img = img
            
        st.session_state.original_size = round(uploaded_file.size/1024, 2)
        with st.expander("📸 Uploaded Image"):
            col1, col2, col3 = st.columns(3)
                
            with col1:
                st.write("👀 **Visual Preview**")
                st.image(img, caption="Original Image", width=256)
            
            with col2:
                st.write("📊 **Image Metadata**")
                st.json({
                    "Filename": uploaded_file.name,
                    "Size": f"{img.size[0]} × {img.size[1]}",
                    "Format": img.format,
                    "Mode": img.mode,
                    "File Size": f"{st.session_state.original_size} KB"
                })

# Home Page
def home():
    st.title("Advanced Image Compression Laboratory")
    st.markdown('<div class="header">Scientific Platform for Compression Research</div>', unsafe_allow_html=True)
    
    st.write("""
    This platform enables rigorous comparison of DCT and DWT compression techniques with:
    - Interactive parameter controls
    - Real-time performance metrics
    - Advanced visualization tools
    - Research documentation space
    """)
    
    # Quick start guide
    with st.expander("Getting Started Guide"):
        st.write("""
        1. **Upload an image** using the file uploader on compression page
        2. **Adjust parameters** for your chosen algorithm
        3. **Run compression** and analyze results
        4. **Compare metrics** in the Analysis Toolkit
        5. **Document findings** in the research notebook
        """)
        
    st.divider()
    
    # Algorithm comparison
    st.subheader("Algorithm Characteristics")
    col1, col2 = st.columns(2)
    dct_path = os.path.join("assets", "DCT.png")
    dwt_path = os.path.join("assets", "DWT.png")
    with col1:
        st.image(dct_path, caption="DCT Block Processing", width=400)
    with col2:
        st.image(dwt_path, caption="DWT Decomposition", width=400)
    
    
    st.markdown("""
    | Feature               | **Compression** | **Best For**       | **Artifacts** | **Complexity** | **Parallelism** |
    |-----------------------|-----------------|--------------------|---------------|----------------|-----------------|
    | **DCT**               | Block-based     | Photographic       | Blocking      | Low            | High            |
    | **DWT**               | Global          | Textured/Medical   | Blurring      | Moderate-High  | Moderate        |
    """)
     
    st.divider()
    st.subheader("Quality Metrics Summary")
    st.markdown("""
    | Metric               | Range/Units       | Strengths                          | Limitations                     | Best Use Case                |
    |----------------------|-------------------|------------------------------------|---------------------------------|------------------------------|
    | **PSNR**             | 0-100 dB          | Simple, fast calculation           | Poor perceptual correlation     | Quick quality checks         |
    | **SSIM**             | 0-1               | Models human perception            | Computationally intensive       | General quality assessment   |
    | **NRMSE**            | 0-1               | Scale-invariant                    | Sensitive to outliers           | Scientific comparisons       |
    | **NMI**              | 0-1               | Robust to intensity distortions    | Complex interpretation          | Medical image registration   |
    | **Compression Ratio**| ≥1 (higher=better)| Intuitive size comparison          | No quality information          | Storage optimization         |
    | **Size Reduction**   | 0%-100%           | Direct space savings percentage    | Depends on original size        | Bandwidth optimization       |
    """)
    
    st.divider()
    st.subheader("Metrics Brief Explanation")
    
    col1, col2 = st.columns(2, border=True)
    
    col1.markdown(r"""
    $$
    \begin{aligned}
    &\color{skyblue}\textbf{Peak Signal-to-Noise Ratio (PSNR)} \\
    &\text{Formula: } \\
    &\text{PSNR} = 10 \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right) \\
    &\text{Range: } [0, \infty) \text{ dB} \\
    &\color{teal}\textbf{Pros:} \\
    &\quad \bullet \text{ Simple computation} \\
    &\quad \bullet \text{ Standardized for quick comparisons} \\
    &\color{brown}\textbf{Cons:} \\
    &\quad \bullet \text{ Poor perceptual correlation} \\
    &\quad \bullet \text{ Sensitive to outliers}
    \end{aligned}
    $$
    """)
    
    col2.markdown(r"""
    $$
    \begin{aligned}
    &\color{skyblue}\textbf{Structural Similarity Index (SSIM)} \\
    &\text{Formula: } \\
    &\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)} \\
    &\text{Range: } [-1, 1] \\
    &\color{teal}\textbf{Pros:} \\
    &\quad \bullet \text{ Models human vision} \\
    &\quad \bullet \text{ Separates luminance/contrast/structure} \\
    &\color{brown}\textbf{Cons:} \\
    &\quad \bullet \text{ Computationally intensive} \\
    &\quad \bullet \text{ Window-size dependent}
    \end{aligned}
    $$
    """)
    
    col1, col2 = st.columns(2, border=True)
    
    col1.markdown(r"""
    $$
    \begin{aligned}
    &\color{skyblue}\textbf{Normalized RMSE} \\
    &\text{Formula: } \\
    &\text{NRMSE} = \frac{\sqrt{\frac{1}{N}\sum_{i=1}^N (x_i - y_i)^2}}{\text{MAX}_I - \text{MIN}_I} \\
    &\text{Range: } [0, 1] \\
    &\color{teal}\textbf{Pros:} \\
    &\quad \bullet \text{ Scale-invariant} \\
    &\quad \bullet \text{ Intuitive interpretation} \\
    &\color{brown}\textbf{Cons:} \\
    &\quad \bullet \text{ Sensitive to extreme errors} \\
    &\quad \bullet \text{ Normalization method affects results}
    \end{aligned}
    $$
    """)
    
    col2.markdown(r"""
    $$
    \begin{aligned}
    &\color{skyblue}\textbf{Normalized Mutual Information} \\
    &\text{Formula: } \\
    &\text{NMI} = \frac{I(X,Y)}{\sqrt{H(X)H(Y)}} \\
    &\text{Range: } [0, 1] \\
    &\color{teal}\textbf{Pros:} \\
    &\quad \bullet \text{ Robust to intensity distortions} \\
    &\quad \bullet \text{ Works for multi-modal data} \\
    &\color{brown}\textbf{Cons:} \\
    &\quad \bullet \text{ Computationally expensive} \\
    &\quad \bullet \text{ Complex interpretation}
    \end{aligned}
    $$
    """)
    
    col1, col2 = st.columns(2, border=True)
    
    col1.markdown(r"""
    $$
    \begin{aligned}
    &\color{skyblue}\textbf{Compression Ratio} \\
    &\text{Formula: } \\
    &\text{CR} = \frac{\text{Original Size}}{\text{Compressed Size}} \\
    &\text{Range: } [1, \infty) \\
    &\color{teal}\textbf{Pros:} \\
    &\quad \bullet \text{ Direct space savings measure} \\
    &\quad \bullet \text{ Hardware-independent} \\
    &\color{brown}\textbf{Cons:} \\
    &\quad \bullet \text{ No quality information} \\
    &\quad \bullet \text{ Content-dependent}
    \end{aligned}
    $$
    """)
    
    col2.markdown(r"""
    $$
    \begin{aligned}
    &\color{skyblue}\textbf{Size Reduction} \\
    &\text{Formula: } \\
    &\text{Reduction} = \left(1 - \frac{\text{Compressed Size}}{\text{Original Size}}\right) \times 100 \\
    &\text{Range: } [0, 100]\% \\
    &\color{teal}\textbf{Pros:} \\
    &\quad \bullet \text{ Intuitive percentage format} \\
    &\quad \bullet \text{ Easy for non-technical users} \\
    &\color{brown}\textbf{Cons:} \\
    &\quad \bullet \text{ Misleading for small files} \\
    &\quad \bullet \text{ No quality indication}
    \end{aligned}
    $$
    """)    

# Plot Interactive chart
def plot_history():
    # Convert to DataFrame with Run numbers
    df = pd.DataFrame([{
        'Run': f"#{i+1}",
        'Algorithm': item['algorithm'],
        **item['metrics']
    } for i, item in enumerate(st.session_state.compression_history)])
    
    # Sidebar controls
    st.sidebar.subheader("Chart Settings")
    metric = st.sidebar.selectbox("Select Metric", df.columns[2:])  # Skip Run and Algorithm
    chart_type = st.sidebar.radio("Chart Type", ["Scatter", "Line", "Bar"], horizontal=True)
    color_theme = st.sidebar.selectbox(
        "Color Scale",
        [None, "ice", "IceFire", "RdBu", "Blues", "Cividis"])
    
    # Main visualization
    st.subheader("Compression Algorithm Performance by Run")

    if chart_type == "Scatter":
        fig = px.scatter(df, x='Run', y=metric,
                        color=metric,
                        color_continuous_scale=color_theme,
                        title=f"{metric} by Run",
                        text=df[metric].round(2),
                        labels={'Run': 'Run Number'},
                        hover_data={'Algorithm': True, metric: ':.2f'})
        fig.update_traces(marker=dict(size=14),
                        textposition='top center')

    elif chart_type == "Line":
        fig = px.line(df, x='Run', y=metric,
                    color="Algorithm",
                    title=f"{metric} Trend",
                    markers=True,
                    labels={'Run': 'Run Number'})
        fig.update_traces(line=dict(width=3))

    else:  # Bar chart
        fig = px.bar(df, x='Run', y=metric,
                    color=metric,
                    color_continuous_scale=color_theme,
                    title=f"{metric} Comparison",
                    text="Algorithm",
                    labels={'Run': 'Run Number'})
        fig.update_traces(textposition='outside')

    # Common layout updates
    fig.update_layout(
        xaxis_title="Run Number",
        yaxis_title=metric,
        hovermode="x unified",
        xaxis={'categoryorder': 'array', 'categoryarray': df['Run']},
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Raw data display
    with st.expander("📊 View Raw Data"):
        st.dataframe(df, use_container_width=True, hide_index=True)

# Analysis Toolkit Page
def toolkit():
    st.title("Analysis Toolkit")
    st.markdown('<div class="header">Result Comparison and Research Notebook</div>', unsafe_allow_html=True)
    
    # Compression history
    st.subheader("Compression History")
    if st.session_state.compression_history:
        with st.container(height=512):
            for i, run in enumerate(st.session_state.compression_history):
                with st.expander(f"Run #{i+1}: {run['algorithm']} Compression"):
                    col1, col2 , col3= st.columns(3)
                    with col1:
                        st.image(run["image"], caption=f"Compressed: {run['compressed_size']} KB")
                    with col2:
                        st.write("**Algorithm Parameters:**")
                        st.json(run["params"])
                    with col3:
                        st.write("**Performance Metrics:**")
                        st.json(run["metrics"])
        
        st.divider()
        plot_history()

    else:
        st.warning(' No records found.', icon="⚠️")
        st.info(' Please upload an image to begin compression.', icon="ℹ️")

    st.divider()
    # Research notebook
    st.subheader("Research Notebook")
    research_notes = st.text_area("Document your observations, hypotheses, and findings", 
                                 height=150,
                                 placeholder="Enter research notes here...",
                                 key="textarea")
    
    if st.button("Save Notes"):
        st.success("Research notes saved!")

# Wavelet Explorer Page
def explorer():
    st.title("Wavelet Coefficient Explorer")
    st.markdown('<div class="header">Multi-resolution Analysis Visualization</div>', unsafe_allow_html=True)
    
    upload_preview()
    
    if st.session_state.original_img is not None:
        # Parameters
        st.sidebar.subheader("Visualization Parameters")
        wavelet = st.sidebar.selectbox("Wavelet Type", 
                                       ["db1", "db2", "db4", "haar", "bior2.2"])
        levels = st.sidebar.slider("Decomposition Levels", 1, 4, 2)
        channel = st.sidebar.radio("Channel", ["Y", "Cr", "Cb"], horizontal=True,
                                    captions=["Luminance", "Chroma Red", "Chroma Blue"])
        cmap = st.sidebar.selectbox("Colormap", ["bone", "cividis", "viridis","Blues", 'Greys'])
        gamma = st.sidebar.slider("Gamma Correction", 0.1, 1.0, 0.3, 0.1)
        mode = st.sidebar.radio( "Visualization Mode", ["Square", "Hierarchical"], horizontal=True)
        
        # Visualization button
        if st.sidebar.button("Generate Visualization"):
            with st.spinner("Processing wavelet decomposition..."):

                # Generate visualization
                fig = coeffs_plot.depict(
                    ycc_img=imgcodecs.rgb2ycc(st.session_state.original_img),
                    chan=channel,
                    mode=mode,
                    wavelet=wavelet,
                    level=levels,
                    cmap=cmap,
                    gamma=gamma
                )

                # Display in Streamlit
                st.pyplot(fig)

# Compression Page
def compression():
    st.title("Compression Analysis")
    st.markdown('<div class="header">Block-based and Wavelet-based Frequency Domain Compression</div>', unsafe_allow_html=True)
    
    upload_preview()
    
    if st.session_state.original_img is not None:
        # Parameters
        st.sidebar.subheader("Algorithm Parameters")
        algo = st.sidebar.selectbox("Compression Algorithm", ['DCT', 'DWT'])
        
        options = ["4:4:4", "4:2:2", "4:2:0"]
        lum_ratio = st.sidebar.pills("Luminance Ratio", options, selection_mode="single", default="4:4:4")
        chroma_ratio = st.sidebar.pills("Chroma Ratio", options, selection_mode="single", default="4:2:2")

        if algo == 'DCT':
            quality = st.sidebar.slider("Quality Factor", 0, 100, 85, 5, key="dct_quality")
            block_size = st.sidebar.selectbox("Block Size", [4, 8, 16], index=1, key="dct_block")
            with st.sidebar.expander("Advanced Options"):
                dct_norm = st.radio("DCT Normalization", ["ortho", "None"])
                padding_mode = st.selectbox("Padding Mode", ["reflect", "edge", "constant"])        
            
        elif algo == 'DWT':
            method = st.sidebar.radio("Compression Method", ["Threshold", "Quantization"], horizontal=True)
            wavelet = st.sidebar.selectbox("Wavelet Type", ["db1", "db2", "db4", "haar", "bior2.2"])
            levels = st.sidebar.slider("Decomposition Levels", 1, 4, 2)

            if method == "Threshold":
                threshold = st.sidebar.slider("Threshold Value", 0, 100, 30, 5)
                threshold_mode = st.sidebar.selectbox("Threshold Mode", ["soft", "hard"])
            else:
                quant_step = st.sidebar.slider("Quantization Step", 0, 100, 20, 5)
        
        juxtapose = st.sidebar.toggle("Juxtaposed Comparison", value=True)
                
        # Compression button
        if st.sidebar.button("Run Compression"):
            with st.spinner("Processing ..."):
                # Call backend compression
                if algo == 'DCT':
                    compressed_ycc = imgcodecs.dct(
                        imgcodecs.rgb2ycc(st.session_state.original_img),
                        lum_ratio=lum_ratio,
                        chr_ratio=chroma_ratio,
                        block_h=block_size,
                        block_w=block_size,
                        quality=quality
                    )
                    st.session_state.compression_history.append({
                        "algorithm": algo,
                        "params": {
                            "quality": quality,
                            "block_size": block_size,
                            "lum_ratio": lum_ratio,
                            "chroma_ratio": chroma_ratio
                        }
                        })
                elif algo == 'DWT':
                    if method == "Threshold":
                        compressed_ycc = imgcodecs.dwt(
                            imgcodecs.rgb2ycc(st.session_state.original_img),
                            lum_ratio=lum_ratio,
                            chr_ratio=chroma_ratio,
                            method='threshold',
                            wavelet=wavelet,
                            level=levels,
                            threshold=threshold,
                            threshold_mode=threshold_mode
                        )
                        st.session_state.compression_history.append({
                            "algorithm": algo,
                            "params": {
                                "method": method,
                                "threshold": threshold,
                                "mode": threshold_mode,
                                "wavelet": wavelet,
                                "levels": levels,
                                "lum_ratio": lum_ratio,
                                "chroma_ratio": chroma_ratio
                            }
                            })
                    else:
                        compressed_ycc = imgcodecs.dwt(
                            imgcodecs.rgb2ycc(st.session_state.original_img),
                            lum_ratio=lum_ratio,
                            chr_ratio=chroma_ratio,
                            method='quantize',
                            wavelet=wavelet,
                            level=levels,
                            quant_step=quant_step
                        )
                        st.session_state.compression_history.append({
                            "algorithm": algo,
                            "params": {
                                "method": method,
                                "quantization": quant_step,
                                "wavelet": wavelet,
                                "levels": levels,
                                "lum_ratio": lum_ratio,
                                "chroma_ratio": chroma_ratio
                            }
                            })
                       
                # Save compressed image    
                compressed_rgb = imgcodecs.ycc2rgb(compressed_ycc)
                st.session_state.compressed_img = compressed_rgb
                temp_path = os.path.join("assets", "temp.jpg")
                Image.fromarray(compressed_rgb).save(temp_path)
                size_in_bytes = os.path.getsize(temp_path)
                st.session_state.compressed_size = round(size_in_bytes/1024, 2)
                
                # Display results
                with st.container(key="sidebyside"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(st.session_state.original_img,
                                caption=f"Original Size: {st.session_state.original_size} KB")
                    with col2:
                        st.image(st.session_state.compressed_img,
                                caption=f"Compressed Size: {st.session_state.compressed_size} KB")
                
                with st.container(key="juxtapose"):
                    if juxtapose:
                        st.caption("Juxtaposed Comparison")
                        image_comparison(
                            img1=st.session_state.original_img,
                            img2=st.session_state.compressed_img,
                            label1="Original", label2="Compressed", width=512, in_memory=True)
                        
                # Calculate metrics
                metrics = imgcodecs.metrics(st.session_state.original_img, compressed_rgb)
                comp_ratio = round(st.session_state.original_size/st.session_state.compressed_size, 2)
                size_reduction = round((1 - 1/comp_ratio) * 100, 2)
                
                # Update history
                st.session_state.compression_history[-1].update({
                    "original_size": st.session_state.original_size,
                    "compressed_size": st.session_state.compressed_size,
                    "image": compressed_rgb,
                })

                metrics.update({
                    "compression_ratio": comp_ratio,
                    "size_reduction": size_reduction
                })
                st.session_state.compression_history[-1].update({
                    "metrics": metrics
                })
                
                if len(st.session_state.compression_history) >= 2:
                    previous_metrics = st.session_state.compression_history[-2]["metrics"]
                    deltas = [round(metrics[key] - previous_metrics[key], 2) for key in metrics]
                else:
                    deltas = [None for key in metrics]
                
                st.divider()
                # Display metrics
                st.subheader("Performance Metrics")
                metric_cols = st.columns(3)
                metric_cols[0].metric("PSNR", f"{metrics['PSNR']} dB", delta=deltas[0])
                metric_cols[1].metric("SSIM", f"{metrics['SSIM']}", delta=deltas[1])
                metric_cols[2].metric("NRMSE", f"{metrics['NRMSE']}", delta=deltas[2],
                                      delta_color="inverse")
                
                metric_cols = st.columns(3)
                metric_cols[0].metric("NMI", f"{metrics['NMI']}", delta=deltas[3])
                metric_cols[1].metric("Compression Ratio", f"{comp_ratio:.2f}:1", delta=deltas[4])
                metric_cols[2].metric("Size Reduction", f"{size_reduction:.2f}%", delta=deltas[5])
                
# Sidebar navigation
st.sidebar.title("Compression Laboratory")
pages = ["Home", "Compression", "Wavelet Explorer", "Analysis Toolkit"]
page = st.sidebar.selectbox("Navigation", pages, key='selbox')

# Home Page
if page == "Home":
    home()

# DCT Compression Page
elif page == "Compression":
    compression()
              
# Wavelet Explorer Page
elif page == "Wavelet Explorer":
    explorer()
                
# Analysis Toolkit
elif page == "Analysis Toolkit":
    toolkit()
    
# Run the app
if __name__ == "__main__":
    pass