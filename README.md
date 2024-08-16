
# Lane Detection using Python, OpenCV, and NumPy

This project contains a Python script for detecting lane lines in images and videos using OpenCV and NumPy. The project utilizes Canny Edge Detection and Hough Transform to identify lane lines, which are crucial for applications like self-driving cars.







## Features

- **Edge Detection:** The Canny Edge Detector finds edges where the gradient of pixel intensities is high, representing potential lane lines.
- **Region of Interest (ROI):** A mask is applied to the image to focus on the region where lanes are likely to be found (usually the lower half of the image).
- **Hough Transform:** Detects straight lines in the masked image. This method is robust in detecting lines even in noisy environments.
- **Lane Line Averaging:** Detected lines are averaged to create smooth lane boundaries, reducing the noise from individual line detections.

## Prerequisites
- Python 3.9.0 or higher 
- OpenCV
- Matplotlib
- NumPy

## Installation

1. Clone the repository:

```bash
  git clone https://github.com/M-ED/Lane_Detection_using_OpenCV.git
```

2. Create virtual environment using following commands:
```bash
  conda create -n projects_CV python==3.9.0
  conda activate projects_CV
```

3. Install the necessary libraries in requirements file
```bash
   pip install -r requirements.txt
```

4. Run the script
```bash
  python main.py
```


## Acknowledgements

- OpenCV: [https://opencv.org/](https://opencv.org/)




## License

[MIT](https://choosealicense.com/licenses/mit/)


## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)


## Author

- [@mohtadia_naqvi](https://github.com/M-ED)

