import os
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_SERVER_PORT'] = '7860'
os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'

import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import SamProcessor, SamModel
import matplotlib.pyplot as plt
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Oil Spill Detection",
    page_icon="🛢️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    """Load the trained SAM model"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Loading model on {device}...")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, None, None
        
        # Load processor
        st.info("Loading SAM processor...")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        
        # Load model
        st.info("Loading SAM model...")
        model = SamModel.from_pretrained("facebook/sam-vit-base")
        
        st.info("Loading trained weights...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        st.success("Model loaded successfully!")
        return model, processor, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None

def create_grid_points(array_size, grid_size=10):
    """Create a grid of points for prompting"""
    x = np.linspace(0, array_size-1, grid_size)
    y = np.linspace(0, array_size-1, grid_size)
    xv, yv = np.meshgrid(x, y)
    
    xv_list = xv.tolist()
    yv_list = yv.tolist()
    
    input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] 
                    for x_row, y_row in zip(xv_list, yv_list)]
    
    input_points = torch.tensor(input_points).view(1, 1, grid_size*grid_size, 2)
    return input_points

def predict_oil_spill(model, processor, image, device, target_size=(256, 256)):
    """Predict oil spill segmentation on a single image"""
    
    # Resize image
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    pil_image = Image.fromarray(image_resized)
    
    # Use grid of points as prompt
    h, w = target_size
    grid_size = min(h, w) // 25
    input_points = create_grid_points(min(h, w), grid_size)
    
    inputs = processor(pil_image, input_points=input_points, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    
    # Apply sigmoid and threshold
    seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    seg_prob = seg_prob.cpu().numpy().squeeze()
    seg_mask = (seg_prob > 0.5).astype(np.uint8)
    
    # Resize back to original size
    original_h, original_w = image.shape[:2]
    seg_mask_resized = cv2.resize(seg_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    seg_prob_resized = cv2.resize(seg_prob, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    
    return seg_mask_resized, seg_prob_resized

def create_overlay(image, mask, alpha=0.5):
    """Create an overlay of the mask on the original image"""
    overlay = image.copy()
    
    # Create colored mask (red for oil spill)
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [255, 0, 0]  # Red color
    
    # Blend
    overlay = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
    
    return overlay

# Main App
def main():
    st.title("🛢️ Oil Spill Detection System")
    st.markdown("### Upload an image to detect oil spills using AI")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Model path input
    model_path = st.sidebar.text_input(
        "Model Path",
        value="oil_spill_sam_final.pth",
        help="Path to your trained model file"
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust detection sensitivity"
    )
    
    # Overlay alpha
    overlay_alpha = st.sidebar.slider(
        "Overlay Transparency",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    
    # Load model
    with st.spinner("Loading model..."):
        model, processor, device = load_model(model_path)
    
    if model is None:
        st.error("Failed to load model. Please check the model path.")
        return
    
    st.sidebar.success(f"✅ Model loaded on {device.upper()}")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a satellite or aerial image"
    )
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Display original image
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Predict button
        if st.button("🔍 Detect Oil Spill", type="primary"):
            with st.spinner("Analyzing image..."):
                # Make prediction
                seg_mask, seg_prob = predict_oil_spill(model, processor, image, device)
                
                # Apply threshold
                seg_mask_thresholded = (seg_prob > confidence_threshold).astype(np.uint8)
                
                # Calculate statistics
                total_pixels = seg_mask_thresholded.size
                oil_pixels = np.sum(seg_mask_thresholded)
                oil_percentage = (oil_pixels / total_pixels) * 100
                
                # Display results
                with col2:
                    st.subheader("Probability Map")
                    fig, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(seg_prob, cmap='hot')
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    st.pyplot(fig)
                    plt.close()
                
                with col3:
                    st.subheader("Detection Result")
                    overlay = create_overlay(image, seg_mask_thresholded, alpha=overlay_alpha)
                    st.image(overlay, use_container_width=True)
                
                # Display metrics
                st.markdown("---")
                st.subheader("📊 Detection Statistics")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric(
                        label="Oil Spill Coverage",
                        value=f"{oil_percentage:.2f}%"
                    )
                
                with metric_col2:
                    st.metric(
                        label="Affected Pixels",
                        value=f"{oil_pixels:,}"
                    )
                
                with metric_col3:
                    status = "⚠️ Oil Detected" if oil_percentage > 1 else "✅ Clean"
                    st.metric(
                        label="Status",
                        value=status
                    )
                
                # Download section
                st.markdown("---")
                st.subheader("💾 Download Results")
                
                download_col1, download_col2 = st.columns(2)
                
                with download_col1:
                    # Save mask
                    mask_img = Image.fromarray((seg_mask_thresholded * 255).astype(np.uint8))
                    buf = BytesIO()
                    mask_img.save(buf, format="PNG")
                    st.download_button(
                        label="Download Binary Mask",
                        data=buf.getvalue(),
                        file_name="oil_spill_mask.png",
                        mime="image/png"
                    )
                
                with download_col2:
                    # Save overlay
                    overlay_img = Image.fromarray(overlay)
                    buf2 = BytesIO()
                    overlay_img.save(buf2, format="PNG")
                    st.download_button(
                        label="Download Overlay Image",
                        data=buf2.getvalue(),
                        file_name="oil_spill_overlay.png",
                        mime="image/png"
                    )
    
    else:
        st.info("👆 Please upload an image to get started")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🛢️ Oil Spill Detection System | Powered by SAM (Segment Anything Model)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
