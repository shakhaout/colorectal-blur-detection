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



