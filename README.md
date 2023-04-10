[MobileOne: An Improved One millisecond Mobile Backbone](https://arxiv.org/abs/2206.04040)

### Installation

```
conda create -n PyTorch python=3.8
conda activate PyTorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install opencv-python==4.5.5.64
pip install tqdm
```

### Note

* The default training configuration is for `mobile_one-s0`
* The test results including accuracy, params and FLOP are obtained by using fused model

### Parameters and FLOPS

```
Number of parameters: 2078504
Time per operator type:
        15.0684 ms.    91.0851%. Conv
        1.20933 ms.     7.3101%. Relu
       0.242441 ms.     1.4655%. FC
      0.0117301 ms.  0.0709057%. AveragePool
     0.00421935 ms.   0.025505%. Reshape
     0.00261659 ms.  0.0158167%. Gather
     0.00200163 ms.  0.0120994%. ExpandDims
     0.00170158 ms.  0.0102857%. Concat
      0.0007769 ms. 0.00469618%. Shape
        16.5432 ms in Total
FLOP per operator type:
       0.548173 GFLOP.    99.6276%. Conv
       0.002049 GFLOP.   0.372395%. FC
              0 GFLOP.          0%. Concat
              0 GFLOP.          0%. Relu
       0.550222 GFLOP in Total
Feature Memory Read per operator type:
        19.7686 MB.    50.6551%. Conv
        15.1532 MB.    38.8285%. Relu
         4.1041 MB.    10.5164%. FC
        1.2e-05 MB. 3.07489e-05%. Concat
        39.0258 MB in Total
Feature Memory Written per operator type:
        15.1532 MB.    49.9934%. Conv
        15.1532 MB.    49.9934%. Relu
          0.004 MB.  0.0131968%. FC
          8e-06 MB. 2.63937e-05%. Concat
        30.3103 MB in Total
Parameter Memory per operator type:
         4.1801 MB.    50.4837%. Conv
            4.1 MB.    49.5163%. FC
              0 MB.          0%. Concat
              0 MB.          0%. Relu
         8.2801 MB in Total
```

### Train

* Configure your `IMAGENET` dataset path in `main.py` for training
* Run `bash main.sh $ --train` for training, `$` is number of GPUs

### Test

* Configure your `IMAGENET` path in `main.py` for testing
* Run `python main.py --test` for testing

### Results

|    Version     | Epochs | Top-1 Acc | Top-5 Acc | Params (M) | FLOP (G) |                                                                          Download |
|:--------------:|:------:|----------:|----------:|-----------:|---------:|----------------------------------------------------------------------------------:|
| mobile_one-s0  |  300   |         - |         - |       2.08 |    0.550 |                                                                                 - |
| mobile_one-s0* |  300   |      71.4 |      89.9 |       2.08 |    0.550 | [model](https://github.com/jahongir7174/MobileOne/releases/download/v0.0.1/s0.pt) |
| mobile_one-s1* |  300   |      75.8 |      92.8 |       4.76 |    1.650 | [model](https://github.com/jahongir7174/MobileOne/releases/download/v0.0.1/s1.pt) |
| mobile_one-s2* |  300   |      77.4 |      93.2 |       7.80 |    2.596 | [model](https://github.com/jahongir7174/MobileOne/releases/download/v0.0.1/s2.pt) |
| mobile_one-s3* |  300   |      77.9 |      93.9 |      10.07 |    3.791 | [model](https://github.com/jahongir7174/MobileOne/releases/download/v0.0.1/s3.pt) |
| mobile_one-s4* |  300   |      79.3 |      94.4 |      14.83 |    5.960 | [model](https://github.com/jahongir7174/MobileOne/releases/download/v0.0.1/s4.pt) |

* `*` means that weights are ported from original repo, see reference

#### Reference

* https://github.com/apple/ml-mobileone
