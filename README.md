# YOLOv7 for MASTIFF

This repo is an adaptation of the [original YOLOv7 repo](https://github.com/WongKinYiu/yolov7) to classify signals in a spectrum waterfall plot. The dataset can be generated using the datagenerator from the [MASTIFF Gym](https://github.com/vtnsi/mastiff).


Here are some sample images after training a YOLOv7x model to detect all signals (left) and zigbee only (right).
<img align="right" width="400" src="https://github.com/user-attachments/assets/69d11c96-3212-4ad5-bc56-f69945d2167a">
|![step_9000](https://github.com/user-attachments/assets/8be9105d-2473-4889-95a1-87b724ee6984)

## Installation

1. Setup Python virtual environment

To simplify dependencies on the system, set up a python virtual environment.
This allows users to locally install dependencies and isolate versions of
packages between projects without affecting other projects. You may create the
virtual environment in any directory, and it does *not* need to be in the Git
repository path. For this example, we will use `~/env/yolo`:

    virtualenv --system-site-packages ~/env/yolo
    source ~/env/yolo/bin/activate

2. Install dependencies (in the python virtual environment)

    cd /path/to/repository
    pip install -r requirements.txt


## Training

Dataset Generation

Follow instructions on the datagenerator in the [MASTIFF Gym](https://github.com/vtnsi/mastiff)
Once you have installed the MASTIFF Gym and generated a dataset (the larger the better, 10k total images recommended), you need to move them to a file structure consistent with the [original YOLOv7 repo](https://github.com/WongKinYiu/yolov7):
data/
    train/
        images/
        labels/
    val/
        images/
        labels/
    test/
        images/
        labels/
The recommended data split is 7:2:1, i.e. if you have a total of 10k images and labels, 7k:2k:1k for train:val:test
Remember to delete `train.cache` and `val.cache` files from your dataset if/when you change it before training again.

Single GPU training
The following will train a YOLOv7 model on the dataset specified in `data/signals.yaml`.

The yaml files consistent with the [original YOLOv7 repo](https://github.com/WongKinYiu/yolov7) were updated in this fork to point to your dataset directory after running the datagenerator from [MASTIFF Gym](https://github.com/vtnsi/mastiff).

 # train yolov7x model
``` shell    
python train.py --device 0 --batch-size 20 --data data/signals.yaml --img 640 640 --cfg cfg/training/yolov7x.yaml --weights '' -name yolov7 --hyp data/hyp.scratch.custom.signal.yaml --epochs 5

Note that `--weights` is an empty string, learning from a random weights initialization, instead of transfer learning. `--data` is the location of your dataset, so change the paths in `signals.yaml` accordingly. `--hyp` was changed slightly to stop flipping, since signals do not look different when flipped. However, hyp.scratch.custom.signal.yaml can be changed to suit future training. `--batch-size` and `--device` will need to be changed to suit the capabilities of the device you are running on.

## Transfer learning

Instead of training from scratch, you can also transfer learn from a weight checkpoint. For example, you can transfer learn starting with weights of the models linked in the [MASTIFF Gym](https://github.com/vtnsi/mastiff).

``` shell
# finetune from checkpoint weights
python train.py --device 0 --batch-size 20 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights '<path/to/checkpoint>.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml
```
By default the training script saves the best model every 5 epochs.
For a full list of options in the script, run `python3 train.py --help`

## Testing

This returns the results for the whole test set

``` shell
python test.py --data data/signals.yaml --img 640 --batch 16 --conf 0.001 --iou 0.45 --device 0 --weights '<path/to/weights>.pt' --name yolov7_640_val
```


## Transfer learning

If you already have a base model and want to further train the weights from where you left off, use transfer learning. The only difference is the --weights parameter now points to the checkpoint .pt file.

Single GPU finetuning for custom dataset

``` shell
# finetune from checkpoint weights
python train.py --device 0 --batch-size 20 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights '<path/to/checkpoint>.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml
```

## Inference

This will return detections on an image or folder of images. The results will be saved under `runs/detect/`

On image:
``` shell
    python detect.py --weights <path/to/final>.pt --conf 0.25 --img-size 640 --source <path/to/images>.png
```

## Citation

```
@inproceedings{wang2023yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

```
@article{wang2023designing,
  title={Designing Network Design Strategies Through Gradient Path Analysis},
  author={Wang, Chien-Yao and Liao, Hong-Yuan Mark and Yeh, I-Hau},
  journal={Journal of Information Science and Engineering},
  year={2023}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>
