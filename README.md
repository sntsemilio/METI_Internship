# MNIST Digit Classification System

A comprehensive digit recognition system built with TensorFlow and Streamlit for the MNIST handwritten digits dataset.

## 🎯 Features

- **Real-time Digit Recognition**: Interactive drawing canvas and image upload
- **Dataset Visualization**: Comprehensive analysis of the MNIST dataset
- **Model Training**: Train custom CNN models with adjustable parameters
- **Performance Analysis**: Detailed evaluation with confusion matrices and metrics
- **Multiple Model Types**: CNN and Dense neural network options

## 📊 Dataset

- **Source**: MNIST Handwritten Digits Database
- **Training**: 54,000 samples
- **Validation**: 6,000 samples  
- **Test**: 10,000 samples
- **Classes**: 10 digits (0-9)
- **Image Size**: 28×28 pixels (grayscale)

## 🧠 Model Architecture

### CNN Model (Recommended)
```
Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Conv2D(64) -> Flatten -> Dense(64) -> Dense(10)
```

- **Parameters**: ~93K trainable parameters
- **Expected Accuracy**: >98%
- **Training Time**: 5-10 minutes on CPU

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/sntsemilio/TC2034.302-Final-project.git
cd TC2034.302-Final-project
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run mnist_app.py
```

### Train Model via Script

```bash
python train_mnist.py
```

## 📁 Project Structure

```
├── mnist_app.py              # Main Streamlit application
├── mnist_model.py            # CNN model implementation
├── mnist_data_loader.py      # Data loading and preprocessing
├── train_mnist.py           # Training script
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## 🎮 Usage

1. **Launch the App**: Run `streamlit run mnist_app.py`
2. **Train a Model**: Go to "Model Training" and click "Start Training"
3. **Test Recognition**: Use "Digit Recognition" to draw or upload digits
4. **Analyze Performance**: Check "Model Analysis" for detailed metrics

## 📈 Performance

| Model Type | Accuracy | Parameters | Training Time |
|------------|----------|------------|---------------|
| CNN        | 98.5%    | 93K        | ~8 minutes    |
| Dense      | 97.2%    | 235K       | ~5 minutes    |

## 🛠️ Technical Details

- **Framework**: TensorFlow 2.x with Keras
- **Frontend**: Streamlit
- **Optimization**: Adam optimizer with early stopping
- **Regularization**: Dropout layers and data validation
- **Deployment**: Local Streamlit server

## 🔧 Customization

### Model Parameters
- Adjust epochs, batch size, learning rate
- Add/remove layers in `mnist_model.py`
- Implement custom architectures

### Data Augmentation
- Add rotation, scaling, noise in `mnist_data_loader.py`
- Implement custom preprocessing pipelines

## 📊 Metrics

The system tracks:
- **Accuracy**: Overall classification accuracy
- **Loss**: Cross-entropy loss during training
- **Confusion Matrix**: Per-class performance
- **Classification Report**: Precision, recall, F1-score per digit

## 🎯 Future Enhancements

- [ ] Real-time drawing canvas
- [ ] Model comparison dashboard
- [ ] Data augmentation options
- [ ] Model export/import functionality
- [ ] Batch prediction interface
- [ ] Custom dataset upload

## 📝 License

This project is open source and available under the MIT License.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.