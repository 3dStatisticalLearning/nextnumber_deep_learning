#  Advanced Number Sequence Prediction API

A production-ready deep learning system for predicting the next row in numerical sequences using Transformer neural networks. Built with PyTorch, FastAPI, and designed for easy deployment on Render.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Running Locally](#running-locally)
- [API Documentation](#api-documentation)
- [Deployment to Render](#deployment-to-render)
- [Configuration & Tuning](#configuration--tuning)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

##  Overview

This project implements a state-of-the-art Transformer-based neural network to predict the next row of numbers in a sequence. Unlike traditional classification approaches, this system uses **regression with temporal patterns**, learning from multiple previous rows to capture complex sequential relationships.

### Use Cases

- **Number Pattern Analysis**: Identify trends and patterns in numerical sequences
- **Statistical Forecasting**: Predict next values based on historical data
- **Time Series Prediction**: Analyze temporal dependencies in ordered data
- **Data Science Research**: Experiment with sequence-to-sequence learning

### Key Highlights

-  **Transformer Architecture**: State-of-the-art attention mechanism for sequence learning
-  **Temporal Context**: Uses 10 previous rows (configurable) for predictions
-  **Automatic Range Detection**: Learns valid number ranges from training data
-  **Production Ready**: Optimized for deployment with proper error handling
-  **Training Monitoring**: Built-in validation, early stopping, and loss tracking

---

##  Features

### Model Features

- **Advanced Transformer Architecture**
  - Multi-head self-attention mechanism
  - Positional encoding for sequence awareness
  - Layer normalization and residual connections
  - Dropout for regularization

- **Intelligent Training**
  - Train/validation split (85/15)
  - Early stopping to prevent overfitting
  - Learning rate scheduling
  - Gradient clipping for stability
  - Automatic checkpoint saving

- **Robust Predictions**
  - Data normalization/denormalization
  - Integer rounding with valid range clamping
  - Batch processing support
  - Multi-step future predictions

### API Features

- **RESTful Endpoints**
  - Single row prediction
  - Multiple row prediction
  - Health check and diagnostics
  - Automatic model configuration loading

- **Error Handling**
  - Input validation
  - Detailed error messages
  - File format verification
  - Dimension checking

---

##  Architecture

### Model Architecture

```
Input Sequence (10 × 20 numbers)
        ↓
[Linear Projection] → Hidden Dim: 256
        ↓
[Positional Encoding]
        ↓
[Transformer Encoder]
├── 4 Layers
├── 8 Attention Heads
└── Feed-forward Dim: 1024
        ↓
[Output Layers]
├── Linear (256 → 256)
├── LayerNorm + ReLU
├── Linear (256 → 128)
├── ReLU
└── Linear (128 → 20)
        ↓
Predicted Next Row (20 numbers)
```

### System Flow

```
Training Phase:
Excel Data → Preprocessing → Model Training → Save Artifacts
                                ↓
                        [model.pth, stats, config]

Prediction Phase:
Upload File → Load Artifacts → Normalize → Model Inference → Denormalize → Integer Output
```

---

##  Prerequisites

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: ~500MB for dependencies and model files

### Required Libraries

All dependencies are listed in `requirements.txt`:

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pandas==2.1.3
numpy==1.26.2
torch==2.1.1
python-multipart==0.0.6
openpyxl==3.1.2
gunicorn==21.2.0
scikit-learn==1.3.2
```

---

##  Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/number-prediction-api.git
cd number-prediction-api
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; import fastapi; print('Installation successful!')"
```

---

##  Training the Model

### Prepare Your Data

Your training data should be an Excel file (`.xlsx` or `.xls`) with:
- **No headers** (all rows contain numerical data)
- **Consistent columns** (same number of columns in every row)
- **Numerical values only** (integers or floats)

**Example data format:**
```
3   13  14  15  16  25  26  32  39  42  45  47  53  58  62  65  68  70  ...
2   6   11  16  20  23  27  29  35  38  41  46  47  51  55  60  64  67  ...
2   7   8   9   14  15  18  22  29  36  37  50  51  54  58  61  66  69  ...
```

### Configure Training

Edit `train.py` line 20 to point to your data file:

```python
CONFIG = {
    "data_path": "your_data.xlsx",  # or full path: r"C:\path\to\your\data.xlsx"
    "sequence_length": 10,
    "hidden_dim": 256,
    "num_heads": 8,
    "num_layers": 4,
    "dropout": 0.2,
    "learning_rate": 0.0001,
    "batch_size": 32,
    "epochs": 200,
    "patience": 20,
}
```

### Run Training

```bash
python train.py
```

### Training Output

You'll see progress like this:

```
Loading dataset...
Dataset shape: (7199, 20)
Value range: [1.0, 70.0]
Mean: 35.52, Std: 20.20

Training samples: 6110
Validation samples: 1079
Using device: cpu

Starting training...
✓ Model saved with validation loss: 1.003786
Epoch [5/200], Train Loss: 0.999830, Val Loss: 1.003329
✓ Model saved with validation loss: 1.003329
...
Early stopping triggered at epoch 29

Training completed!
Best validation loss: 1.003179
```

### Generated Files

After training, these files will be created:

-  `model.pth` - Trained model weights (required for deployment)
-  `model_stats.json` - Data statistics for normalization (required)
-  `model_config.json` - Model configuration (required)
-  `train_losses.npy` - Training loss history (optional)
-  `val_losses.npy` - Validation loss history (optional)

---

##  Running Locally

### Start the API Server

**Command Line (Recommended):**
```bash
python app.py
```

**Or using Uvicorn directly:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Access the API

- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### Test with Example Request

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/predict/" \
  -F "file=@your_test_data.xlsx"
```

**Using Python:**
```python
import requests

url = "http://localhost:8000/predict/"
files = {"file": open("your_test_data.xlsx", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

---

## 📡 API Documentation

### Endpoints

#### 1. Root Endpoint
```http
GET /
```

**Response:**
```json
{
  "message": "Improved Number Prediction API",
  "version": "2.0",
  "model_info": {
    "sequence_length": 10,
    "input_dimension": 20,
    "value_range": "[1, 70]",
    "device": "cpu"
  }
}
```

#### 2. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "stats": {
    "min_value": 1.0,
    "max_value": 70.0,
    "mean": 35.52,
    "std": 20.20
  }
}
```

#### 3. Single Prediction
```http
POST /predict/
Content-Type: multipart/form-data
```

**Parameters:**
- `file` (required): Excel file with historical data

**Requirements:**
- File must be `.xlsx` or `.xls`
- Must have same number of columns as training data
- Must contain at least 10 rows (sequence_length)

**Response:**
```json
{
  "success": true,
  "prediction": [3, 7, 12, 19, 24, 28, 31, 35, 42, 45, 48, 51, 55, 58, 62, 67, 71, 78, 83, 89],
  "metadata": {
    "input_rows": 150,
    "sequence_used": 10,
    "value_range": "[1, 70]",
    "model_device": "cpu"
  }
}
```

#### 4. Multiple Predictions
```http
POST /predict-multiple/?n_predictions=5
Content-Type: multipart/form-data
```

**Parameters:**
- `file` (required): Excel file with historical data
- `n_predictions` (optional): Number of future rows to predict (1-10, default: 1)

**Response:**
```json
{
  "success": true,
  "predictions": [
    [3, 7, 12, 19, 24, 28, 31, 35, 42, 45, 48, 51, 55, 58, 62, 67, 71, 78, 83, 89],
    [5, 9, 15, 21, 26, 30, 34, 38, 44, 47, 50, 53, 57, 60, 64, 69, 73, 80, 85, 90],
    [2, 11, 18, 23, 27, 32, 36, 40, 46, 49, 52, 55, 59, 62, 66, 71, 75, 82, 87, 92]
  ],
  "count": 3,
  "value_range": "[1, 70]"
}
```

### Error Responses

**400 Bad Request:**
```json
{
  "detail": "Expected 20 columns, got 15"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Prediction error: Invalid input format"
}
```

---

##  Deployment to Render

### Prerequisites

1. Create a [Render account](https://render.com/)
2. Push your code to GitHub/GitLab
3. Ensure all required files are in the repository

### Required Files for Deployment

```
your-repo/
├── app.py                   FastAPI application
├── requirements.txt         Dependencies
├── model.pth               Trained model (IMPORTANT!)
├── model_stats.json        Data statistics
├── model_config.json       Model configuration
└── README.md               This file
```

### Deployment Steps

#### Option 1: Using Render Dashboard

1. **Go to Render Dashboard**
   - Visit https://dashboard.render.com/
   - Click "New +" → "Web Service"

2. **Connect Repository**
   - Select "Build and deploy from a Git repository"
   - Connect your GitHub/GitLab account
   - Select your repository

3. **Configure Service**
   - **Name**: `number-prediction-api` (or your choice)
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave blank (unless your files are in a subdirectory)
   - **Runtime**: `Python 3`

4. **Build Settings**
   - **Build Command**: 
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**: 
     ```bash
     gunicorn app:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
     ```

5. **Instance Type**
   - Free tier: Good for testing (will sleep after inactivity)
   - Starter ($7/month): Recommended for production

6. **Environment Variables** (Optional)
   - Add `PYTHON_VERSION=3.10.0` if you want to specify Python version

7. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes for initial deployment
   - Your API will be available at: `https://your-service-name.onrender.com`

#### Option 2: Using render.yaml (Infrastructure as Code)

Create a `render.yaml` file in your repository root:

```yaml
services:
  - type: web
    name: number-prediction-api
    runtime: python
    plan: starter  # or 'free' for testing
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
    envVars:
      - key: PYTHON_VERSION
        value: "3.10.0"
```

Then in Render:
1. Click "New +" → "Blueprint"
2. Connect repository
3. Render will automatically detect and use `render.yaml`

### Post-Deployment

#### Verify Deployment

Test your deployed API:

```bash
# Health check
curl https://your-service-name.onrender.com/health

# Test prediction
curl -X POST "https://your-service-name.onrender.com/predict/" \
  -F "file=@test_data.xlsx"
```

#### Monitor Logs

- Go to your service dashboard on Render
- Click "Logs" tab to view real-time logs
- Check for any errors or warnings

#### Performance Monitoring

- Monitor response times in Render dashboard
- Check memory usage (upgrade plan if needed)
- Review error rates

### Common Deployment Issues

**Issue: "Model file not found"**
- Solution: Ensure `model.pth` is committed to Git (check `.gitignore`)
- Large files: Use Git LFS for files >100MB

**Issue: "Memory limit exceeded"**
- Solution: Upgrade to a paid plan with more RAM
- Or: Reduce model size by training with fewer layers/hidden dimensions

**Issue: "Application timeout"**
- Solution: Add health check endpoint configuration in Render
- Or: Ensure your start command is correct

---

##  Configuration & Tuning

### Training Configuration

Edit the `CONFIG` dictionary in `train.py`:

```python
CONFIG = {
    # Data
    "data_path": "your_data.xlsx",
    
    # Model Architecture
    "sequence_length": 10,      # Number of previous rows to use
    "hidden_dim": 256,          # Hidden layer size (128, 256, 512)
    "num_heads": 8,             # Attention heads (4, 8, 16)
    "num_layers": 4,            # Transformer layers (2, 4, 6, 8)
    "dropout": 0.2,             # Dropout rate (0.1-0.3)
    
    # Training
    "learning_rate": 0.0001,    # Learning rate (0.00001-0.001)
    "batch_size": 32,           # Batch size (16, 32, 64)
    "epochs": 200,              # Maximum epochs
    "patience": 20,             # Early stopping patience
}
```

### Hyperparameter Guidelines

| Parameter | Small Dataset | Large Dataset | High Accuracy |
|-----------|---------------|---------------|---------------|
| sequence_length | 5 | 15 | 20 |
| hidden_dim | 128 | 512 | 512 |
| num_layers | 2 | 6 | 8 |
| num_heads | 4 | 8 | 16 |
| batch_size | 16 | 64 | 32 |
| learning_rate | 0.0005 | 0.0001 | 0.00005 |

### When to Retrain

Retrain your model when:
-  You have significantly more data (2x+ rows)
-  Data patterns have changed
-  Validation loss hasn't improved
-  You want to experiment with different architectures

---

##  Performance Tips

### Training Performance

1. **Use GPU if available**
   - PyTorch automatically uses CUDA if available
   - Training can be 10-50x faster on GPU

2. **Optimize batch size**
   - Larger batches = faster training but more memory
   - Start with 32, increase to 64 or 128 if memory allows

3. **Monitor validation loss**
   - If val_loss > train_loss significantly: reduce model complexity
   - If both losses are high: increase model capacity or train longer

4. **Data quality matters**
   - More data = better predictions (aim for 500+ rows)
   - Consistent patterns = better learning

### Prediction Performance

1. **Use smaller models for production**
   - Reduce hidden_dim or num_layers if speed is critical
   - Test accuracy vs speed trade-off

2. **Batch predictions**
   - If predicting for multiple files, batch requests
   - Use async processing for concurrent requests

3. **Cache frequently used predictions**
   - Implement Redis/Memcached for common inputs
   - Reduce redundant computations

---

##  Troubleshooting

### Training Issues

**Problem: Loss not decreasing**
```
Solution:
- Lower learning rate (try 0.00001)
- Check data quality (ensure numerical values are valid)
- Increase model capacity (more layers/hidden_dim)
- Train longer (increase epochs)
```

**Problem: Validation loss increasing while training loss decreases**
```
Solution:
- Overfitting detected!
- Increase dropout (0.3-0.4)
- Reduce model size
- Get more training data
- Enable early stopping (already included)
```

**Problem: Training very slow**
```
Solution:
- Reduce batch_size (try 16)
- Reduce model size temporarily
- Use GPU if available
- Reduce sequence_length
```

### API Issues

**Problem: "FileNotFoundError: model.pth"**
```
Solution:
- Run train.py first
- Check current directory with os.getcwd()
- Ensure model.pth is in same folder as app.py
```

**Problem: "Expected X columns, got Y"**
```
Solution:
- Test file must have same columns as training data
- Check training data column count
- Verify Excel file format
```

**Problem: Predictions seem random**
```
Solution:
- Model may need more training data
- Check if data has learnable patterns
- Retrain with higher capacity model
- Verify data normalization is working
```

### Deployment Issues

**Problem: Render deployment fails**
```
Solution:
- Check Render logs for specific error
- Verify all required files are in repository
- Ensure requirements.txt has correct versions
- Check start command syntax
```

**Problem: API responds slowly**
```
Solution:
- Upgrade Render plan (more CPU/RAM)
- Reduce model size
- Enable caching
- Use batch predictions
```

---

##  Project Structure

```
number-prediction-api/
│
├── app.py                      # FastAPI application (main API)
├── train.py                    # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── model.pth                   # Trained model weights (generated)
├── model_stats.json           # Data statistics (generated)
├── model_config.json          # Model configuration (generated)
│
├── train_losses.npy           # Training history (optional)
├── val_losses.npy             # Validation history (optional)
│
├── .gitignore                 # Git ignore file
└── render.yaml                # Render deployment config (optional)
```

### File Descriptions

| File | Purpose | Required for Deployment |
|------|---------|------------------------|
| `app.py` | Main API application |  Yes |
| `train.py` | Training script |  No (only for retraining) |
| `requirements.txt` | Dependencies |  Yes |
| `model.pth` | Trained model |  Yes |
| `model_stats.json` | Normalization stats |  Yes |
| `model_config.json` | Model architecture |  Yes |
| `*.npy` files | Training history |  No (for analysis only) |

---

##  Contributing

Contributions are welcome! Here's how you can help:

### Reporting Issues

1. Check existing issues first
2. Include detailed description
3. Provide reproduction steps
4. Share error messages and logs

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to functions
- Include unit tests for new features
- Update README if adding new functionality

---

##  Example Use Cases

### Case 1: Lottery Number Analysis

```python
# Train on historical lottery draws
# Configure: sequence_length=20 for more context
# Model learns statistical patterns (not guaranteed prediction!)
```

### Case 2: Time Series Forecasting

```python
# Convert time series to rows
# Each row = features at a time point
# Predict next time point values
```

### Case 3: Pattern Recognition Research

```python
# Experiment with different architectures
# Analyze which patterns the model learns
# Compare with statistical methods
```

---

##  Performance Benchmarks

Based on a dataset of 7,199 rows with 20 columns:

| Metric | Value |
|--------|-------|
| Training Time | ~5-10 minutes (CPU) |
| Final Validation Loss | ~1.003 |
| Prediction Time | <100ms per row |
| Model Size | ~15 MB |
| Memory Usage | ~500 MB |

---

##  Security & Privacy

### Data Privacy

- No data is stored on the server
- Predictions are computed in real-time
- Uploaded files are processed in memory only
- No logging of sensitive prediction data

### API Security

- Input validation on all endpoints
- File type verification
- Size limits on uploads (handled by FastAPI)
- Error messages don't expose system details

### Recommendations for Production

- Add authentication (API keys, OAuth)
- Implement rate limiting
- Use HTTPS (Render provides this automatically)
- Monitor for abuse
- Add request logging for debugging

---

##  License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

##  Disclaimer

**Important Notice:**

This tool is designed for pattern recognition and statistical analysis. It should NOT be used for:

-  Gambling or betting decisions
-  Financial predictions or investment advice
-  Any situation where accurate prediction is critical

**Key Points:**

- Predictions are based on historical patterns
- No guarantee of accuracy for truly random events
- Past patterns don't guarantee future outcomes
- Use responsibly and at your own risk

For truly random processes (like lottery draws), no model can reliably predict future outcomes. This tool is best suited for data with underlying patterns or trends.

---

##  Support & Contact

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/number-prediction-api/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/number-prediction-api/discussions)
- **Email**: your.email@example.com

### Useful Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Render Documentation](https://render.com/docs)
- [Transformer Models Guide](https://jalammar.github.io/illustrated-transformer/)

---

##  Acknowledgments

- PyTorch team for the excellent deep learning framework
- FastAPI for the modern web framework
- Render for easy deployment platform
- The open-source community for various tools and libraries

---

##  Roadmap

### Upcoming Features

- [ ] Web interface for easy file upload
- [ ] Confidence scores for predictions
- [ ] Multiple model ensemble
- [ ] Real-time training monitoring dashboard
- [ ] Docker containerization
- [ ] Support for CSV files
- [ ] Batch processing endpoint
- [ ] Model versioning
- [ ] A/B testing framework

### Future Enhancements

- [ ] GPU optimization guide
- [ ] Advanced hyperparameter tuning
- [ ] Transfer learning capabilities
- [ ] Explainability features (attention visualization)
- [ ] Auto-scaling configuration

---

##  Changelog

### Version 2.0 (Current)
- Switched from classification to regression
- Added sequence learning (10-row context)
- Implemented proper normalization
- Added early stopping and validation
- Automatic range detection
- Multiple prediction endpoint
- Comprehensive error handling

### Version 1.0
- Initial release with basic classification model

---

<div align="center">

**Made with  for the Data Science Community**

[⬆ Back to Top](#-advanced-number-sequence-prediction-api)

</div># nextnumber_deep_learning
