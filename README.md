# pytorch-hymenoptera
Pytorch CV project template for deep learning researchers, take hymenoptera dataset as an example.

# Requirement
* Pytorch 0.3+
* Torch vision 0.1.9+
* Python 2.7

# Dataset
This template takes the Pytorch official tutorial's hymenoptera dataset as an example, to run the example:
* Create a directory to store the dataset

```
cd pytorch-hymenoptera/

mkdir dataset
```
* Update the config.json file in **/pytorch-hymenoptera/data_loader**
```
{
    "data_dir": "../dataset/hymenoptera_data",
    "batch_size": 4,
    "num_workers": 4
}
```
* Download the dataset form [here](https://download.pytorch.org/tutorial/hymenoptera_data.zip), put it into dataset directory and unzip it
```
 cd pytorch-hymenoptera/
 cd dataset/
 unzip hymenoptera_data.zip
```
* Train and test

# Example
* Train the model, if you use the default model settings(including seed in config.json), you will get 94.118% val accuracy
```
python train.py
```
* Test the model, in this example, we still use the validation set to test our model(don't do this in practice), so you will get the test accuracy near 94.118% too. Note that you should choose the checkpoint file manually
```
python test.py --checkpoint logging/checkpoint.pth
```
* Plot the curve
```
python plot.py
```
This will produce two image files: accuracy.jpg, loss.jpg

![accuracy](https://github.com/Kexiii/pytorch-hymenoptera/blob/master/accuracy.jpg)
![loss](https://github.com/Kexiii/pytorch-hymenoptera/blob/master/loss.jpg)

* Resume from previous work
```
python train.py --resume logging/checkpoint_to_resume.pth
```
