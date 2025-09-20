# Medical Image Classifier

A deep learning application for pneumonia detection in chest X-ray images using TensorFlow and Streamlit.

## Project Overview

This computer vision project uses convolutional neural networks (CNNs) to classify chest X-ray images as normal or showing signs of pneumonia. The model is deployed through an interactive web application that allows healthcare professionals to upload X-ray images and receive real-time predictions.

## Features

- **Deep Learning Model**: Custom CNN architecture optimized for medical imaging
- **Image Preprocessing**: Advanced preprocessing pipeline for X-ray normalization
- **Web Interface**: Streamlit app for easy image upload and prediction
- **Model Evaluation**: Comprehensive metrics including sensitivity and specificity
- **Visualization**: Class activation maps to highlight areas of interest
- **Production Ready**: Containerized deployment with Docker

## Tech Stack

- **Deep Learning**: TensorFlow 2.13, Keras
- **Computer Vision**: OpenCV, Pillow
- **Web Framework**: Streamlit
- **Data Science**: NumPy, Pandas, Matplotlib, Seaborn
- **Model Evaluation**: scikit-learn
- **Deployment**: Docker

## Project Structure

```
medical-image-classifier/
├── data/
│   ├── train/              # Training images
│   ├── test/               # Test images
│   └── val/                # Validation images
├── models/
│   ├── best_model.h5       # Trained model weights
│   └── model_architecture.json
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA and visualization
│   ├── 02_model_training.ipynb      # CNN training process
│   └── 03_model_evaluation.ipynb   # Performance analysis
├── src/
│   ├── data_preprocessing.py        # Image preprocessing pipeline
│   ├── model_training.py           # CNN training script
│   ├── prediction.py               # Inference functions
│   └── utils.py                    # Helper functions
├── web_app/
│   └── streamlit_app.py            # Web interface
├── requirements.txt               # Python dependencies
└── Dockerfile                    # Container configuration
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sunnynguyen-ai/medical-image-classifier.git
cd medical-image-classifier
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python src/model_training.py
```

### Running the Web Application
```bash
streamlit run web_app/streamlit_app.py
```
Visit `http://localhost:8501` to access the interface.

### Making Predictions
```python
from src.prediction import predict_pneumonia

# Load and predict
result = predict_pneumonia('path/to/xray.jpg')
print(f"Prediction: {result['class']} (Confidence: {result['confidence']:.2f})")
```

## Model Performance

- **Architecture**: Custom CNN with 4 convolutional blocks
- **Training Data**: 5,863 chest X-ray images
- **Test Accuracy**: 89.3%
- **Sensitivity**: 92.1% (detecting pneumonia cases)
- **Specificity**: 86.7% (identifying normal cases)
- **AUC-ROC**: 0.94

## Dataset

This project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle, containing:

- **Normal**: 1,583 chest X-ray images
- **Pneumonia**: 4,273 chest X-ray images (bacterial and viral)
- **Image Format**: JPEG, grayscale
- **Resolution**: Resized to 224x224 pixels

## Key Features

### Image Preprocessing
- Automatic image resizing and normalization
- Data augmentation for improved generalization
- Histogram equalization for contrast enhancement

### Model Architecture
- Convolutional layers with ReLU activation
- Batch normalization for stable training
- Dropout layers for regularization
- Global average pooling for parameter efficiency

### Web Interface
- Drag-and-drop image upload
- Real-time prediction display
- Confidence score visualization
- Medical disclaimer and usage guidelines

## Results and Insights

- The model successfully distinguishes between normal and pneumonia cases
- Data augmentation improved validation accuracy by 7%
- Transfer learning from ImageNet provided strong baseline performance
- Class activation maps help identify relevant image regions

## Future Improvements

- [ ] Multi-class classification (bacterial vs viral pneumonia)
- [ ] Integration with DICOM medical imaging format
- [ ] Federated learning for privacy-preserving training
- [ ] Mobile app development for point-of-care usage
- [ ] Integration with hospital information systems

## Medical Disclaimer

This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

This project is open source and available under the MIT License.

## Contact

**Sunny Nguyen**
- GitHub: [@sunnynguyen-ai](https://github.com/sunnynguyen-ai)
- Email: sunny.nguyen@onimail.com
- Website: [sunnyinspires.com](https://sunnyinspires.com)

## Acknowledgments

- Dataset provided by Kermany et al. via Kaggle
- Inspired by recent advances in medical AI research
- Built with the TensorFlow and Streamlit communities
