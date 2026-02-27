#  Mini Transformer from Scratch (IMDB Sentiment Analysis)

##  Overview

This project implements a **Transformer Encoder from scratch using PyTorch** for binary sentiment classification on the IMDB movie reviews dataset.

The model is trained for movie review classification (Positive / Negative) and later deployed as a **Flask web application** for real-time inference.

---

# Model Architecture

The Transformer model is implemented completely from scratch without using HuggingFace or pre-built transformer modules.

### Architecture Components:

- 🔹 Token Embedding  
- 🔹 Positional Encoding  
- 🔹 Masked Multi-Head Self Attention  
- 🔹 Feed Forward Network  
- 🔹 Residual Connections + LayerNorm  
- 🔹 Mean Pooling  
- 🔹 Final Classification Head  

---

#  Dataset

- **Dataset:** IMDB Movie Reviews  
- **Task:** Binary Sentiment Classification  
- **Classes:**
  - POSITIVE  
  - NEGATIVE  

---

#  Training Details

 Parameter                 Value 
 Optimizer                 Adam 
 Loss Function         CrossEntropyLoss
 Batch Size                 32 
 Epochs                     25
 Max Vocabulary Size       1000 
 Max Sequence Length       100

---

#  Project Structure

```
Sentiment_Analysis_using_Transformers/
│
├── app.py                  # Flask application
├── train.py                # Training script
├── mini_transformer.pth    # Trained model
├── requirements.txt
│
├── Dataset/
│   ├── dataset.py
│   └── imdb_train.csv
│
├── src/
│   └── pipeline/
│       └── model.py
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

---

# 🏋️ How To Train The Model

## 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt

```

## 2️⃣ Run Training

```bash
python train.py
```

After training completes, the model will be saved as:

```
mini_transformer.pth
```

---

# 🌐 Flask Deployment (Web Application)

After training, the model is deployed using **Flask** to serve predictions through a web interface.

---

## How To Run Flask App

### 1 Start Flask Server

```bash
python app.py
```

---

## 🌍 Local Deployment URL

Once the server starts, open your browser and go to:

```
http://127.0.0.1:5000
```

This runs the application locally on your machine.

--- 

# Tech Stack

- Python
- PyTorch
- Flask
- HTML
- CSS
- Pandas

---

#  Future Improvements

- Deploy to Render / AWS / Railway  
- Add Docker support  
- Add REST API endpoint  
- Add attention visualization  
- Improve model with multiple transformer layers  

---

# 👨‍💻 Author

**Shanmukha Pasumarthi**  
