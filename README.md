🔬 Blood Cell Detection using YOLOv10
This project uses YOLOv10 to detect RBCs, WBCs, and Platelets from blood smear images. The model runs in a Streamlit web app and is deployed on Hugging Face Spaces.

📌 Features
Detects RBCs, WBCs, and Platelets in blood sample images.
Displays bounding boxes with different colors for each cell type.
Confidence score for each detected object.
User-friendly Streamlit web app interface.
🚀 How to Run Locally
Clone the repository:
bash
Copy
Edit
git clone https://github.com/yourusername/blood-cell-detection.git
cd blood-cell-detection
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:
bash
Copy
Edit
streamlit run app.py
🌍 Deploy on Hugging Face
This app is deployed on Hugging Face Spaces. Try it here:
🔗 Live Demo

📦 Requirements
Python 3.8+
torch, ultralytics, opencv-python, streamlit, pillow, numpy, pandas
📜 License
This project is open-source under the MIT License.
