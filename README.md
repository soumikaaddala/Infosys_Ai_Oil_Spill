🛰️ SAM-Based Oil Spill Segmentation

This project fine-tunes Meta’s Segment Anything Model (SAM) to detect and segment oil spills from satellite images.
It also includes a Streamlit web app (app.py) that allows users to upload an image and view predicted segmentation results.

📁 Project Files

sam.ipynb – Jupyter notebook used to train and test the SAM model on the Oil Spill dataset.

app.py – Streamlit app to visualize segmentation results.

README.md – Description of the project (this file).

requirements.txt – list of dependencies.

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/soumikaaddala/Infosys_Ai_Oil_Spill.git

cd Infosys_Ai_Oil_Spill


2️⃣ Install dependencies

You can install all required libraries using:

pip install streamlit torch torchvision transformers monai opencv-python matplotlib datasets patchify

🚀 How to Run
▶️ Run the Jupyter Notebook

Open and execute sam.ipynb in Google Colab or Jupyter Notebook to train and test the model.

▶️ Run the Streamlit App

Once the model (oil_spill_sam_final.pth) is saved:

streamlit run app.py


Then open the provided local URL (usually http://localhost:8501) in your browser.

📊 Model Details

Base model: facebook/sam-vit-base

Libraries used: PyTorch, MONAI, Transformers

Dataset: Oil Spill Dataset – Zenodo

Loss Function: Dice + Cross-Entropy Loss
