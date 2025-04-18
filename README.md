# Dog Breed Predictor 🐕

An AI-powered dog breed identification system that can recognize 120 different dog breeds in real-time using your webcam. Built with TensorFlow, MobileNetV2, and OpenCV.

## Features 

- Real-time breed detection through webcam
- Support for 120 different dog breeds
- Pre-trained deep learning model using MobileNetV2 architecture
- High accuracy breed identification
- Easy-to-use interface

## Demo 🎥

![Demo Image](https://github.com/user-attachments/assets/f62dd0c5-2135-4040-94ef-9f0e435d23c8)

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Dog-breed-predictor.git
cd Dog-breed-predictor
```

2. Download the model:
[Download pre-trained model](https://drive.google.com/drive/folders/1P6EXf6EyAfhYCc92VOQgeaSJlB9MLs5l?usp=drive_link)

## Usage 💻

1. Make sure your webcam is connected
2. Run the webcam detection script:
```bash
python opencv-image.py
```
3. Point your webcam at a dog
4. Press 'q' to quit

## Model Details 🧠

- **Architecture**: MobileNetV2
- **Training Dataset**: Custom dataset with 120 dog breeds
- **Input Size**: 224x224 pixels
- **Output**: Probability distribution across 120 breed classes

## Project Structure 📁

```
├── opencv-image.py      # Main script for webcam detection
├── dog_vision.ipynb     # Training notebook
├── README.md           # Project documentation
```

## Requirements 📋

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- TensorFlow Hub

## Training 🚀

The model was trained using transfer learning on MobileNetV2 architecture. For training details, check `dog_vision.ipynb`.

## Contributing 🤝

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License 📝

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgments 🙏

- TensorFlow team for MobileNetV2
- Stanford Dogs Dataset
- OpenCV community

