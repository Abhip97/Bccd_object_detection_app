# **🔬 Blood Cell Detection using YOLOv10**  

This project uses **YOLOv10** to detect **RBCs, WBCs, and Platelets** from blood smear images. The model runs in a **Streamlit web app** and is deployed on **Hugging Face Spaces**.  

## **📌 Features**  
- Detects **RBCs, WBCs, and Platelets** in blood sample images.  
- Displays bounding boxes with different colors for each cell type.  
- Confidence score for each detected object.  
- User-friendly **Streamlit web app** interface.  

## **🚀 How to Run Locally**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/blood-cell-detection.git
   cd blood-cell-detection
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```

## **🌍 Deploy on Hugging Face**  
This app is deployed on Hugging Face Spaces. Try it here:  
🔗 **[Live Demo](https://huggingface.co/spaces/your-space-name)**  

## **📦 Requirements**  
- Python 3.8+  
- `torch`, `ultralytics`, `opencv-python`, `streamlit`, `pillow`, `numpy`, `pandas`  

## **📜 License**  
This project is **open-source** under the MIT License.  
