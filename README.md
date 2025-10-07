# Pneumonia Detection with Machine Learning

A deep learning project that uses Convolutional Neural Networks (CNNs) to detect pneumonia from chest X-ray images. The model is built using TensorFlow/Keras.

## Project Overview

This project implements a binary classifier that can identify the presence of pneumonia in chest X-ray images. The model uses a custom CNN architecture and includes:

- Data preprocessing and augmentation
- Model training with validation
- Performance monitoring and visualization
- Inference script for making predictions on new images

## Dataset

The dataset contains chest X-ray images (anterior-posterior) organized into two categories:
- NORMAL: Normal chest X-rays
- PNEUMONIA: Chest X-rays showing pneumonia

The data is split into training, validation, and test sets.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wtruong0/ML-Pneumonia-Imaging.git
cd ML-Pneumonia-Imaging
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

TODO:
- Image Resizing Instructions
- Training Instructions
- Inference Instructions

## Model Architecture

Description TBD

## Results

The model achieves:
- Training accuracy: (x results)
- Validation accuracy: (x results)
- Test set accuracy: (x results)

Training history plots are saved in the model output directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- Dataset source: [Kaggle Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Programmed by [**Will Truong**](https://www.linkedin.com/in/truongw/)
