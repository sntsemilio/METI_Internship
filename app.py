import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

# Handle TensorFlow imports with proper error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    
    # Configure TensorFlow to avoid GPU issues
    tf.config.set_visible_devices([], 'GPU')
    
    # Suppress TensorFlow warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
except ImportError as e:
    st.error(f"TensorFlow import failed: {e}")
    TF_AVAILABLE = False
except Exception as e:
    st.warning(f"TensorFlow configuration issue: {e}. Using CPU fallback.")
    TF_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="MNIST Digit Classification System",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 0.25rem solid #1f77b4;
}
.prediction-digit {
    background-color: #e3f2fd;
    color: #1565c0;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 0.25rem solid #1565c0;
    font-size: 2rem;
    text-align: center;
}
.prediction-confidence {
    background-color: #e8f5e8;
    color: #2e7d32;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 0.25rem solid #2e7d32;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'tf_session' not in st.session_state:
    st.session_state.tf_session = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

# MNIST Model Components (same as Colab)
if TF_AVAILABLE:
    def create_mnist_model():
        """Create the same CNN model from Colab"""
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

@st.cache_resource
def create_tensorflow_session():
    """Create and configure TensorFlow session"""
    if not TF_AVAILABLE:
        return None
    
    try:
        # Configure TensorFlow for CPU usage
        tf.config.set_visible_devices([], 'GPU')
        
        # Create session configuration
        session_config = {
            "allow_soft_placement": True,
            "log_device_placement": False
        }
        
        return session_config
        
    except Exception as e:
        st.error(f"Failed to create TensorFlow session: {e}")
        return None

@st.cache_data
def load_mnist_dataset():
    """Load and preprocess MNIST dataset (same as Colab)"""
    try:
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape for CNN (add channel dimension)
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Create validation split
        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
        
    except Exception as e:
        st.error(f"Could not load MNIST dataset: {e}")
        return None, None, None

def load_trained_model():
    """Load the trained model from Colab"""
    model = None
    training_history = None
    model_results = None
    
    if TF_AVAILABLE:
        try:
            # Try to load the saved model
            model = keras.models.load_model('final_mnist_model.h5')
            st.success("✅ Trained model loaded from 'final_mnist_model.h5'")
        except:
            try:
                model = keras.models.load_model('best_mnist_model.h5')
                st.success("✅ Trained model loaded from 'best_mnist_model.h5'")
            except:
                st.warning("⚠️ No trained model found. You can train a new model in the Training section.")
                # Create a new untrained model
                model = create_mnist_model()
        
        # Try to load training history
        try:
            with open('training_history.pkl', 'rb') as f:
                training_history = pickle.load(f)
            st.info("📊 Training history loaded successfully")
        except:
            st.info("ℹ️ No training history found")
        
        # Try to load model results
        try:
            with open('model_summary.pkl', 'rb') as f:
                model_results = pickle.load(f)
            st.info("📈 Model results loaded successfully")
        except:
            st.info("ℹ️ No model results found")
    
    return model, training_history, model_results

def predict_with_model(image, model):
    """Make predictions using the trained model"""
    if not TF_AVAILABLE or model is None:
        return predict_digit_mock(image)
    
    try:
        # Ensure image is in correct format
        if len(image.shape) == 2:
            image = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 3:
            image = image.reshape(1, *image.shape)
        
        # Make prediction
        with tf.device('/CPU:0'):  # Force CPU usage
            predictions = model.predict(image, verbose=0)
        
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        probabilities = predictions[0]
        
        return predicted_class, confidence, probabilities
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return predict_digit_mock(image)

def predict_digit_mock(image):
    """Mock prediction function for fallback"""
    # Simple mock prediction based on random with some logic
    np.random.seed(int(np.sum(image) * 1000) % 2**32)
    
    # Create realistic probabilities
    probabilities = np.random.dirichlet(np.ones(10) * 0.5)
    
    # Make one digit more likely based on image characteristics
    dominant_digit = np.random.randint(0, 10)
    probabilities[dominant_digit] *= 3
    probabilities = probabilities / probabilities.sum()
    
    predicted_class = np.argmax(probabilities)
    confidence = np.max(probabilities)
    
    return predicted_class, confidence, probabilities

# Load model and data at startup
if not st.session_state.model_loaded:
    with st.spinner("🔄 Loading trained model and dataset..."):
        # Create TensorFlow session
        tf_session = create_tensorflow_session()
        st.session_state.tf_session = tf_session
        
        # Load trained model
        model, training_history, model_results = load_trained_model()
        st.session_state.model = model
        st.session_state.training_history = training_history
        st.session_state.model_results = model_results
        
        # Load dataset
        dataset = load_mnist_dataset()
        st.session_state.dataset = dataset
        
        st.session_state.model_loaded = True

# Sidebar navigation
st.sidebar.title("🔢 Navigation")

# Display model status in sidebar
if TF_AVAILABLE and st.session_state.model:
    if st.session_state.training_history:
        st.sidebar.success("✅ Trained Model Loaded")
    else:
        st.sidebar.warning("⚠️ Untrained Model")
else:
    st.sidebar.warning("⚠️ Mock Mode")

# Model info in sidebar
if st.session_state.model_results:
    st.sidebar.markdown("### 📊 Model Stats")
    st.sidebar.metric("Test Accuracy", f"{st.session_state.model_results['test_accuracy']:.1%}")
    st.sidebar.metric("Parameters", f"{st.session_state.model_results['total_parameters']:,}")

page = st.sidebar.selectbox(
    "Choose a page",
    ["🏠 Home", "🎯 Digit Recognition", "📊 Dataset Visualization", "🧠 Model Training", "📈 Model Analysis"]
)

# Main content based on page selection
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🔢 MNIST Digit Classification System</h1>', unsafe_allow_html=True)
    
    # Model status
    if TF_AVAILABLE and st.session_state.model:
        if st.session_state.training_history:
            st.success("🚀 **Trained Model Active** - Real-time digit recognition enabled")
        else:
            st.info("📝 **Model Ready** - Train the model to enable full functionality")
    else:
        st.warning("⚠️ **Mock Mode** - TensorFlow not available, using simulated predictions")
    
    st.markdown("""
    ## Welcome to the MNIST Digit Classification System
    
    This application demonstrates handwritten digit recognition using a Convolutional Neural Network.
    The system is based on the model trained in Google Colab with comprehensive evaluation capabilities.
    
    ### 🎯 Key Features:
    - **Real-time Recognition**: Test the trained CNN model on digit images
    - **Dataset Analysis**: Comprehensive visualization of the MNIST training data
    - **Model Training**: Train new models with interactive monitoring
    - **Performance Analysis**: Detailed evaluation with confusion matrices and metrics
    
    ### 📊 Dataset Information:
    - **Source**: MNIST Handwritten Digits Database
    - **Training Samples**: 54,000 images (with validation split)
    - **Test Samples**: 10,000 images
    - **Classes**: 10 digits (0-9)
    - **Image Size**: 28×28 pixels (grayscale)
    
    ### 🧠 Model Architecture (from Colab):
    ```
    Conv2D(32, 3×3) → MaxPool(2×2)
    Conv2D(64, 3×3) → MaxPool(2×2)
    Conv2D(64, 3×3)
    Flatten → Dense(64) → Dropout(0.5) → Dense(10)
    ```
    
    ### 📈 Performance Highlights:
    """)
    
    if st.session_state.model_results:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Accuracy", f"{st.session_state.model_results['test_accuracy']:.1%}")
        with col2:
            st.metric("Parameters", f"{st.session_state.model_results['total_parameters']:,}")
        with col3:
            if 'epochs_trained' in st.session_state.model_results:
                st.metric("Epochs Trained", st.session_state.model_results['epochs_trained'])
            else:
                st.metric("Model Type", "CNN")
        with col4:
            if 'best_val_accuracy' in st.session_state.model_results:
                st.metric("Best Val Accuracy", f"{st.session_state.model_results['best_val_accuracy']:.1%}")
            else:
                st.metric("Status", "Ready")
    else:
        st.info("Train the model to see performance metrics!")
    
    st.markdown("""
    ### 🔧 Technical Implementation:
    - **Framework**: TensorFlow 2.x with Keras
    - **Architecture**: Convolutional Neural Network optimized for MNIST
    - **Training**: Adam optimizer with early stopping and learning rate reduction
    - **Deployment**: Streamlit with real-time inference capabilities
    
    Use the sidebar to navigate through different sections of the application.
    """)

elif page == "🎯 Digit Recognition":
    st.markdown('<h1 class="main-header">🎯 Digit Recognition Interface</h1>', unsafe_allow_html=True)
    
    # Display session status
    if TF_AVAILABLE and st.session_state.model:
        if st.session_state.training_history:
            st.success("🔥 **Trained Model Active** - Using real CNN inference from Colab training")
        else:
            st.info("📝 **Untrained Model** - Train the model for best results")
    else:
        st.warning("⚠️ **Mock Mode** - Simulated predictions (TensorFlow unavailable)")
    
    st.markdown("### Test digit recognition with the trained CNN model")
    
    # Image input options
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Image Input")
        
        input_method = st.radio(
            "Choose input method:",
            ["Random Test Sample", "Upload Image", "Sample Patterns"]
        )
        
        image_to_predict = None
        true_label = None
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader("Upload a digit image:", type=['png', 'jpg', 'jpeg'])
            if uploaded_file is not None:
                from PIL import Image
                import cv2
                
                # Process uploaded image
                image = Image.open(uploaded_file).convert('L')
                image_array = np.array(image)
                
                # Resize to 28x28 and normalize
                image_resized = cv2.resize(image_array, (28, 28))
                image_normalized = image_resized.astype('float32') / 255.0
                
                # Invert if needed (MNIST digits are white on black)
                if np.mean(image_normalized) > 0.5:
                    image_normalized = 1.0 - image_normalized
                
                # Display the processed image
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
                ax1.imshow(image_array, cmap='gray')
                ax1.set_title('Original')
                ax1.axis('off')
                ax2.imshow(image_normalized, cmap='gray')
                ax2.set_title('Processed (28×28)')
                ax2.axis('off')
                st.pyplot(fig)
                
                image_to_predict = image_normalized
        
        elif input_method == "Random Test Sample":
            if st.button("🎲 Get Random Sample") and st.session_state.dataset:
                (x_train, y_train), (x_val, y_val), (x_test, y_test) = st.session_state.dataset
                
                # Select random sample
                idx = np.random.randint(0, len(x_test))
                image_to_predict = x_test[idx].squeeze()
                true_label = y_test[idx]
                
                # Display the test image
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(image_to_predict, cmap='gray')
                ax.set_title(f'True Label: {true_label}')
                ax.axis('off')
                st.pyplot(fig)
        
        elif input_method == "Sample Patterns":
            st.info("Select a sample digit pattern to test:")
            
            # Create sample digit patterns
            sample_patterns = {
                "Zero (0)": np.array([
                    [0,0,0,1,1,1,1,0,0,0],
                    [0,0,1,1,1,1,1,1,0,0],
                    [0,1,1,1,0,0,1,1,1,0],
                    [0,1,1,0,0,0,0,1,1,0],
                    [1,1,1,0,0,0,0,1,1,1],
                    [1,1,0,0,0,0,0,0,1,1],
                    [1,1,0,0,0,0,0,0,1,1],
                    [1,1,1,0,0,0,0,1,1,1],
                    [0,1,1,0,0,0,0,1,1,0],
                    [0,1,1,1,0,0,1,1,1,0],
                    [0,0,1,1,1,1,1,1,0,0],
                    [0,0,0,1,1,1,1,0,0,0]
                ]),
                "One (1)": np.array([
                    [0,0,0,0,1,1,0,0,0,0],
                    [0,0,0,1,1,1,0,0,0,0],
                    [0,0,1,1,1,1,0,0,0,0],
                    [0,0,0,0,1,1,0,0,0,0],
                    [0,0,0,0,1,1,0,0,0,0],
                    [0,0,0,0,1,1,0,0,0,0],
                    [0,0,0,0,1,1,0,0,0,0],
                    [0,0,0,0,1,1,0,0,0,0],
                    [0,0,0,0,1,1,0,0,0,0],
                    [0,0,0,0,1,1,0,0,0,0],
                    [0,1,1,1,1,1,1,1,1,0],
                    [0,1,1,1,1,1,1,1,1,0]
                ])
            }
            
            selected_pattern = st.selectbox("Choose pattern:", list(sample_patterns.keys()))
            
            if st.button("🎨 Use Selected Pattern"):
                # Create 28x28 version
                pattern = sample_patterns[selected_pattern]
                # Resize to 28x28
                import cv2
                pattern_28 = cv2.resize(pattern.astype(np.float32), (28, 28))
                
                image_to_predict = pattern_28
                st.image(pattern_28, caption=f"Generated {selected_pattern}", width=200)
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if image_to_predict is not None:
            # Make prediction
            if st.session_state.model:
                prediction, confidence, probabilities = predict_with_model(
                    image_to_predict, st.session_state.model
                )
            else:
                prediction, confidence, probabilities = predict_digit_mock(image_to_predict)
            
            # Display main prediction
            st.markdown(f"""
            <div class="prediction-digit">
                Predicted Digit: <strong>{prediction}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="prediction-confidence">
                Confidence: <strong>{confidence:.1%}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Show prediction with true label if available
            if true_label is not None:
                if prediction == true_label:
                    st.success(f"✅ **Correct!** Model predicted: {prediction}, True label: {true_label}")
                else:
                    st.error(f"❌ **Incorrect!** Model predicted: {prediction}, True label: {true_label}")
            
            # Visualization of probabilities
            fig = go.Figure()
            
            colors = ['red' if i == prediction else 'skyblue' for i in range(10)]
            
            fig.add_trace(go.Bar(
                x=list(range(10)),
                y=probabilities,
                marker_color=colors,
                text=[f"{p:.3f}" for p in probabilities],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Prediction Probabilities",
                xaxis_title="Digit",
                yaxis_title="Probability",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Top 3 predictions
            st.markdown("### 🏆 Top 3 Predictions")
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            
            for i, idx in enumerate(top_3_indices):
                rank = i + 1
                prob = probabilities[idx]
                st.write(f"{rank}. Digit **{idx}** - {prob:.1%}")
        
        else:
            st.info("Please provide an image to analyze using the input methods on the left.")
    
    # Model information
    if st.session_state.model_results:
        st.markdown("### 🧠 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Accuracy", f"{st.session_state.model_results['test_accuracy']:.1%}")
        with col2:
            st.metric("Total Parameters", f"{st.session_state.model_results['total_parameters']:,}")
        with col3:
            if 'training_date' in st.session_state.model_results:
                st.metric("Training Date", st.session_state.model_results['training_date'][:10])

elif page == "📊 Dataset Visualization":
    st.markdown('<h1 class="main-header">📊 Dataset Visualization</h1>', unsafe_allow_html=True)
    
    if st.session_state.dataset is None or st.session_state.dataset[0] is None:
        st.error("Dataset not loaded!")
        st.stop()
    
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = st.session_state.dataset
    
    # Dataset statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Samples", f"{len(x_train):,}")
    with col2:
        st.metric("Validation Samples", f"{len(x_val):,}")
    with col3:
        st.metric("Test Samples", f"{len(x_test):,}")
    
    # Sample visualization
    st.markdown("### 📸 Sample Images from Training Data")
    
    if st.button("🔄 Show Random Samples"):
        fig, axes = plt.subplots(4, 5, figsize=(12, 10))
        fig.suptitle('Random MNIST Training Samples', fontsize=16)
        
        # Select random samples
        indices = np.random.choice(len(x_train), 20, replace=False)
        
        for i, idx in enumerate(indices):
            row = i // 5
            col = i % 5
            
            # Remove channel dimension for visualization
            img = x_train[idx].squeeze()
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f'Label: {y_train[idx]}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Class distribution
    st.markdown("### 📊 Class Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        class_counts = np.bincount(y_train)
        
        fig = px.bar(
            x=list(range(10)),
            y=class_counts,
            title="Training Set Class Distribution",
            labels={'x': 'Digit', 'y': 'Count'},
            color=class_counts,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        test_counts = np.bincount(y_test)
        
        fig = px.bar(
            x=list(range(10)),
            y=test_counts,
            title="Test Set Class Distribution",
            labels={'x': 'Digit', 'y': 'Count'},
            color=test_counts,
            color_continuous_scale='plasma'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Mean digit visualization
    st.markdown("### 🔍 Average Digit Images")
    
    # Calculate mean images for each digit
    mean_images = []
    for digit in range(10):
        digit_images = x_train[y_train == digit]
        mean_image = np.mean(digit_images, axis=0).squeeze()
        mean_images.append(mean_image)
    
    # Display mean images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Average Images for Each Digit', fontsize=16)
    
    for i in range(10):
        row = i // 5
        col = i % 5
        axes[row, col].imshow(mean_images[i], cmap='gray')
        axes[row, col].set_title(f'Digit {i}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Dataset statistics table
    st.markdown("### 📋 Detailed Statistics")
    
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    
    stats_data = {
        'Digit': list(range(10)),
        'Train Count': [int(count) for count in train_counts],
        'Test Count': [int(count) for count in test_counts],
        'Train %': [f"{count/len(y_train)*100:.1f}%" for count in train_counts],
        'Test %': [f"{count/len(y_test)*100:.1f}%" for count in test_counts],
        'Mean Intensity': [f"{np.mean(x_train[y_train == digit]):.3f}" for digit in range(10)]
    }
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True)

elif page == "🧠 Model Training":
    st.markdown('<h1 class="main-header">🧠 Model Training Interface</h1>', unsafe_allow_html=True)
    
    if st.session_state.dataset is None or st.session_state.dataset[0] is None:
        st.error("Dataset not loaded!")
        st.stop()
    
    st.markdown("""
    ### 🎯 Training the MNIST CNN Model
    
    This section allows you to train the same model architecture used in the Colab notebook.
    """)
    
    # Show current model status
    if st.session_state.training_history:
        st.success("✅ Model already trained! You can retrain with different parameters below.")
        
        # Show training history if available
        if st.checkbox("📈 Show Training History"):
            history = st.session_state.training_history
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot accuracy
            ax1.plot(history['accuracy'], label='Training Accuracy', marker='o')
            ax1.plot(history['val_accuracy'], label='Validation Accuracy', marker='s')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot loss
            ax2.plot(history['loss'], label='Training Loss', marker='o')
            ax2.plot(history['val_loss'], label='Validation Loss', marker='s')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("🆕 No trained model found. Train a new model below.")
    
    # Show TensorFlow training status
    if TF_AVAILABLE:
        st.success("🔥 **TensorFlow Training Available** - Real model training enabled")
    else:
        st.warning("⚠️ **Training Not Available** - TensorFlow not available")
        st.stop()
    
    # Training parameters
    st.markdown("### ⚙️ Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Number of Epochs", 1, 30, 15)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        learning_rate = st.selectbox("Learning Rate", [0.0001, 0.001, 0.01], index=1)
    
    with col2:
        use_early_stopping = st.checkbox("Early Stopping", True)
        use_lr_reduction = st.checkbox("Learning Rate Reduction", True)
        save_model = st.checkbox("Save Trained Model", True)
        
        if use_early_stopping:
            patience = st.slider("Early Stopping Patience", 3, 10, 3)
    
    # Model architecture display
    st.markdown("### 🏗️ CNN Architecture (Same as Colab)")
    st.code("""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)       320       
 max_pooling2d (MaxPooling2  (None, 13, 13, 32)       0         
 conv2d_1 (Conv2D)           (None, 11, 11, 64)       18496     
 max_pooling2d_1 (MaxPoolin  (None, 5, 5, 64)         0         
 conv2d_2 (Conv2D)           (None, 3, 3, 64)         36928     
 flatten (Flatten)           (None, 576)               0         
 dense (Dense)               (None, 64)                36928     
 dropout (Dropout)           (None, 64)                0         
 dense_1 (Dense)             (None, 10)                650       
=================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
    """)
    
    # Training execution
    if st.button("🚀 Start Training", type="primary"):
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = st.session_state.dataset
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Training CNN model..."):
            # Create model
            model = create_mnist_model()
            
            # Update learning rate
            model.optimizer.learning_rate.assign(learning_rate)
            
            # Setup callbacks
            callbacks = []
            
            if use_early_stopping:
                callbacks.append(keras.callbacks.EarlyStopping(
                    patience=patience, restore_best_weights=True, verbose=1
                ))
            
            if use_lr_reduction:
                callbacks.append(keras.callbacks.ReduceLROnPlateau(
                    factor=0.2, patience=2, min_lr=0.0001, verbose=1
                ))
            
            if save_model:
                callbacks.append(keras.callbacks.ModelCheckpoint(
                    'streamlit_trained_model.h5',
                    save_best_only=True, verbose=1
                ))
            
            # Custom callback for progress tracking
            class ProgressCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    
                    status_text.text(f"Epoch {epoch + 1}/{epochs} - "
                                   f"Loss: {logs['loss']:.4f} - "
                                   f"Accuracy: {logs['accuracy']:.4f} - "
                                   f"Val Loss: {logs['val_loss']:.4f} - "
                                   f"Val Accuracy: {logs['val_accuracy']:.4f}")
            
            callbacks.append(ProgressCallback())
            
            # Train the model
            history = model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            progress_bar.progress(1.0)
            status_text.text("Training completed!")
            
            # Store trained model
            st.session_state.model = model
            st.session_state.training_history = history.history
            
            # Evaluate on test set
            test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
            
            # Update model results
            st.session_state.model_results = {
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'total_parameters': model.count_params(),
                'epochs_trained': len(history.history['accuracy']),
                'best_val_accuracy': max(history.history['val_accuracy']),
                'training_date': f"2025-06-22 16:10:51"
            }
            
            st.success("✅ Model training completed successfully!")
            
            # Display training results
            st.markdown("### 📈 Training Results")
            
            # Plot training history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot accuracy
            ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot loss
            ax2.plot(history.history['loss'], label='Training Loss', marker='o')
            ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Final metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Final Train Accuracy", f"{history.history['accuracy'][-1]:.4f}")
            with col2:
                st.metric("Final Val Accuracy", f"{history.history['val_accuracy'][-1]:.4f}")
            with col3:
                st.metric("Test Accuracy", f"{test_accuracy:.4f}")
            with col4:
                st.metric("Test Loss", f"{test_loss:.4f}")

elif page == "📈 Model Analysis":
    st.markdown('<h1 class="main-header">📈 Model Analysis & Performance</h1>', unsafe_allow_html=True)
    
    # Check if we have a trained model
    if not st.session_state.model or not st.session_state.training_history:
        st.warning("⚠️ **No trained model found!** Please train a model first in the Training section.")
        st.stop()
    
    if st.session_state.dataset is None:
        st.error("Dataset not loaded!")
        st.stop()
    
    st.success("🔥 **Analyzing Trained Model** - Real performance metrics from TensorFlow")
    
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = st.session_state.dataset
    model = st.session_state.model
    
    # Overall performance metrics
    st.markdown("### 🎯 Overall Performance")
    
    if st.session_state.model_results:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Accuracy", f"{st.session_state.model_results['test_accuracy']:.1%}")
        with col2:
            st.metric("Test Loss", f"{st.session_state.model_results['test_loss']:.4f}")
        with col3:
            st.metric("Parameters", f"{st.session_state.model_results['total_parameters']:,}")
        with col4:
            error_rate = 1 - st.session_state.model_results['test_accuracy']
            st.metric("Error Rate", f"{error_rate:.1%}")
    
    # Detailed evaluation
    with st.spinner("Evaluating model on test set..."):
        # Make predictions
        y_pred_proba = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Confusion Matrix
    st.markdown("### 🎯 Confusion Matrix")
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Blues',
        title="Confusion Matrix - Trained CNN Model"
    )
    
    fig.update_layout(
        xaxis=dict(title="Predicted Digit", tickvals=list(range(10))),
        yaxis=dict(title="True Digit", tickvals=list(range(10))),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Per-class accuracy
    st.markdown("### 📊 Per-Class Performance")
    
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_acc = px.bar(
            x=list(range(10)),
            y=class_accuracy,
            title="Per-Class Accuracy",
            labels={'x': 'Digit', 'y': 'Accuracy'},
            color=class_accuracy,
            color_continuous_scale='viridis'
        )
        fig_acc.update_layout(height=400)
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        # Show class-wise statistics
        class_stats = []
        for digit in range(10):
            true_positives = cm[digit, digit]
            total_actual = cm[digit, :].sum()
            total_predicted = cm[:, digit].sum()
            
            precision = true_positives / total_predicted if total_predicted > 0 else 0
            recall = true_positives / total_actual if total_actual > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_stats.append({
                'Digit': digit,
                'Precision': f"{precision:.3f}",
                'Recall': f"{recall:.3f}",
                'F1-Score': f"{f1:.3f}",
                'Support': int(total_actual)
            })
        
        stats_df = pd.DataFrame(class_stats)
        st.dataframe(stats_df, use_container_width=True)
    
    # Classification report
    st.markdown("### 📋 Detailed Classification Report")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(4)
    
    # Style the dataframe
    def highlight_metrics(s):
        if s.name in ['precision', 'recall', 'f1-score']:
            return ['background-color: #e8f5e8' if v > 0.95 else '' for v in s]
        return [''] * len(s)
    
    styled_report = report_df.style.apply(highlight_metrics)
    st.dataframe(styled_report, use_container_width=True)
    
    # Error analysis
    st.markdown("### 🔍 Error Analysis")
    
    # Find misclassified examples
    misclassified_mask = y_pred != y_test
    misclassified_indices = np.where(misclassified_mask)[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Misclassified", len(misclassified_indices))
        st.metric("Error Rate", f"{len(misclassified_indices)/len(y_test):.1%}")
    
    with col2:
        st.metric("Correctly Classified", len(y_test) - len(misclassified_indices))
        st.metric("Accuracy", f"{(len(y_test) - len(misclassified_indices))/len(y_test):.1%}")
    
    if len(misclassified_indices) > 0:
        if st.button("🔍 Show Misclassified Examples"):
            num_examples = min(15, len(misclassified_indices))
            selected_indices = np.random.choice(misclassified_indices, num_examples, replace=False)
            
            fig, axes = plt.subplots(3, 5, figsize=(15, 9))
            fig.suptitle('Misclassified Examples from Test Set', fontsize=16)
            
            for i, idx in enumerate(selected_indices):
                row, col = i // 5, i % 5
                
                # Get prediction confidence
                confidence = np.max(y_pred_proba[idx])
                
                axes[row, col].imshow(x_test[idx].squeeze(), cmap='gray')
                axes[row, col].set_title(
                    f'True: {y_test[idx]}, Pred: {y_pred[idx]}\n'
                    f'Conf: {confidence:.2f}', 
                    fontsize=10
                )
                axes[row, col].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.success("🎉 Perfect classification! No misclassified examples found.")
    
    # Model insights
    st.markdown("### 💡 Key Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.success("""
        **Model Achievements:**
        - ✅ High accuracy on balanced MNIST dataset
        - ✅ Efficient CNN architecture with only 93K parameters
        - ✅ Good generalization from training to test set
        - ✅ Consistent performance across all digit classes
        - ✅ Fast inference suitable for real-time applications
        """)
    
    with insights_col2:
        st.info("""
        **Technical Observations:**
        - 🧠 CNN effectively learns spatial features of digits
        - 🧠 MaxPooling provides translation invariance
        - 🧠 Dropout prevents overfitting despite small dataset
        - 🧠 3-layer CNN sufficient for MNIST complexity
        - 🧠 Adam optimizer converges efficiently
        """)

# Footer and Application End
st.markdown("---")

# Technical specifications
st.subheader("🔧 Technical Implementation Details")

with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Framework & Libraries**")
        st.write("• TensorFlow 2.x with Keras")
        st.write("• Streamlit for web interface") 
        st.write("• Plotly for interactive visualizations")
        st.write("• Scikit-learn for evaluation metrics")
        st.write("• NumPy & Matplotlib for data processing")
    
    with col2:
        st.markdown("**Model Architecture**")
        st.write("• 3-layer CNN with MaxPooling")
        st.write("• 32→64→64 feature maps progression")
        st.write("• Dropout regularization (0.5)")
        st.write("• Adam optimizer with learning rate scheduling")
        st.write("• Early stopping for optimal training")

# Final footer with student information
st.markdown(f"""
<div style='text-align: center; color: #6c757d; padding: 2rem; border-top: 1px solid #dee2e6; margin-top: 2rem;'>
    <h4 style='color: #495057; margin-bottom: 1rem;'>TC2034.302 - Final Project</h4>
    <p style='margin: 0.5rem 0;'><strong>Student:</strong> Jose Emilio Gomez Santos (@sntsemilio)</p>
    <p style='margin: 0.5rem 0;'><strong>Project:</strong> MNIST Digit Classification using CNN</p>
    <p style='margin: 0.5rem 0;'><strong>Completed:</strong> 2025-06-22 16:10:51 UTC</p>
    <p style='margin: 0.5rem 0;'><strong>Framework:</strong> TensorFlow 2.x | Streamlit | Google Colab Integration</p>
</div>
""", unsafe_allow_html=True)

# Session cleanup and final status
if st.button("🔚 End Session & Cleanup", type="secondary"):
    st.balloons()
    st.success("✅ Application session completed successfully!")
    st.info("Thank you for exploring the MNIST Digit Classification System.")
    
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    st.rerun()

# Final status display
model_status = "Trained" if st.session_state.training_history else "Ready"
st.markdown(f"""
<div style='position: fixed; bottom: 10px; right: 10px; background-color: rgba(0,0,0,0.8); color: white; padding: 0.5rem; border-radius: 5px; font-size: 0.8rem; z-index: 1000;'>
    🔢 MNIST CNN System | Status: {model_status} | @sntsemilio
</div>
""", unsafe_allow_html=True)