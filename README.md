# Colorectal-Blur-Detection
This research work proposes a training free no reference image blur detection using [Walsh-Hadamard](https://en.wikipedia.org/wiki/Hadamard_transform) transform and [KS-statistic](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) analysis.

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

