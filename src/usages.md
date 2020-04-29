### Requirements
```
Pytorch
Torchvision
Nvidia Apex (only for FP16 training)
numpy
cupy (optional but highly recommended especially for training the model, 10x speed up in data preprocessing comparated with numpy)
cv2 (opencv-python)
glob
h5py
json 
win32api (PythonWin) - Should be installed by default in newest Python versions (Python 3.7 reccomended)
```

### Run the model Generate dataset 
* File: generate_data.py
* Usage example: 
```
python generate_data.py --save_dir tedd1007\training_data
```
* How-to:
  * Set your game in windowed mode
  * Set your game to 1600x900 resolution
  * Move the game window to the top left corner, there should be a blue line of 1 pixel in the left bezel of your
         screen and the window top bar should start in the top bezel of your screen.
  * Play the game! The program will capture your screen and generate the training examples. There will be saved
         as files named "training_dataX.npz" (numpy compressed array). Don't worry if you re-launch this script,
          the program will search for already existing dataset files in the directory and it won't overwrite them.
  * At any moment push Q + E to stop the program.
  
<p align="center">
  <img src="github_images/example_config.png" alt="Setup Example"/>
</p>

### Train the model 
* File: train.py
* Usage example: 
```
python train.py --train_new 
--train_dir tedd1007\training_data\train 
--dev_dir tedd1007\training_data\dev 
--test_dir tedd1007\training_data\test 
--output_dir tedd1007\models 
--batch_size 10 
--num_epochs 5 
--fp16
```
* How-to:
  Train a model using the default hyper parameters, to see a description of the network hyper parameters use 
  "python train.py -h" or check the "train.py" and "model.py" files. train, dev and test directories should contain
   as many files named "training_dataX.npz" as you want. The FP16 flag allows you to use Mixed Precision Training if
   you have a modern Nvidia GPU with Tensor cores (RTX 2000, RTX Titan, Titan V, Tesla V100...), 
   it uses the Nvidia Apex library: https://github.com/NVIDIA/apex.
   The model is VERY memory demanding, as a
   reference I use a batch size of 15 for a RTX 2080 (8GB VRAM) for FP16 training (half the Vram usage than FP32 training) 
   using the default parameters. 
   
 * If you want to continue training from a checkpoint use (Note: The checkpoint will automatically use the same 
 floating point precision (FP16 or FP32) used for training when it was created):
   
 ```
python train.py --continue_training
--train_dir tedd1007\training_data\train 
--dev_dir tedd1007\training_data\dev 
--test_dir tedd1007\training_data\test 
--output_dir tedd1007\models 
--batch_size 10 
--num_epochs 5 
--checkpoint_path tedd1007\checkpoint\epoch1checkpoint.pt
```
   
### Run the model
* File: run_TEDD1104.py
* Pretrained-Models: [See the releases sections](https://github.com/ikergarcia1996/Self-Driving-Car-in-Video-Games/releases/)
* Usage example: 
```
python run_TEDD1104.py --model_dir D:\GTAV-AI\models --show_current_control --fp16
```
Use the FP16 flag if you have an Nvidia GPU with tensor cores (RTX 2000, RTX Titan, Titan V...) 
for a nice speed up (~x2 speed up) and half the VRAM usage. 
Requires the Nvidia Apex library: https://github.com/NVIDIA/apex

* How-to:
  * Set your game in windowed mode
  * Set your game to 1600x900 resolution
  * Move the game window to the top left corner, there should be a blue line of 1 pixel in the left bezel of your
         screen and the window top bar should start in the top bezel of your screen.
  * Let the AI play the game!
  * Push Q + E to exit
  * Push L to see the input images
  * Push and hold J to use to use manual control
          
<p align="center">
  <img src="github_images/example_config.png" alt="Setup Example"/>
</p>
