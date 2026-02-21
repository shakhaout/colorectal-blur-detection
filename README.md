# Colorectal-Blur-Detection
## [No-Reference Blurred Image Detection from Colonoscopy Videos Using Walsh-Hadamard Transform and Kolmogorov Smirnov Test](https://ieeexplore.ieee.org/document/10759253)
Endoscopy is a widely used clinical procedure for the early detection of polyps which may develop into cancers if not treated. Colorectal cancer (CRC) being the third most diagnosed cancer in the world requires colonoscopy for diagnosis. Colonoscopy videos often contain blurry frames due to motion, out -of- focus, water jets, etc. making them prone to miss diagnosis due to information loss in the noisy frames. Image Restoration (IR) techniques such as Gaussian and diffusion-based generative denoisers have become very popular recently and have great prospects for restoring medical images. This paper proposes a simple yet effective training-free no-reference colonoscopy image blur detection technique, that can be used in future research for image denoising. This method applies artificial Gaussian blur to the input image and transforms both input and artificial blurred images from the spatial domain to the frequency domain using Walsh-Hadamard (WH) transform. Kolmogorov Smirnov test (KS-statistic) was used to calculate the difference between two distributions of frequencies assuming a blurred image will have a lower difference than a sharp image. Being training-free and using only a single frequency feature this algorithm can be implemented for any domain of images for detecting blurred images, tested on public Image Quality Assessment (IQA) datasets such as CSIQ and LIVE_II and achieved comparable results.

## Requirements
python 3.11.4\
matplotlib 3.7.1\
numpy 1.24.3\
opencv-python 4.9.0.80\
pandas 1.5.3\
scipy 1.11.4\
seaborn 0.12.2\
tqdm 4.65.0\
yaml 0.2.5


## Instructions to run
* Update the `config` file.
* To detect blurriness in input image use following command,
    ```shell
    python inference.py
    ```
* To run evaluation script on public datasets,
    ```shell
    python evaluate.py
    ```
## Citation
If you find this repo useful in your work or research, please cite:
```
@INPROCEEDINGS{10759253,
  author={Hossain, MD Shakhaout and Ono, Naoaki and Kanaya, Shigehiko and Altaf-Ul-Amin, Md.},
  booktitle={2024 IEEE International Conference on Imaging Systems and Techniques (IST)}, 
  title={No-Reference Blurred Image Detection from Colonoscopy Videos Using Walsh-Hadamard Transform and Kolmogorov Smirnov Test}, 
  year={2024},
  volume={},
  number={},
  pages={1-6},
  keywords={Training;Frequency-domain analysis;Colonoscopy;Estimation;Transforms;Colorectal cancer;Reflection;Real-time systems;Image restoration;Videos;Colonoscopy blur image detection;medical image;no reference blur assessment;walsh-hadamard;frequency domain;kolmogorov smirnov test},
  doi={10.1109/IST63414.2024.10759253}}
```

