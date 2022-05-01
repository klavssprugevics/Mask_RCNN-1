import os
import sys
import time
import shutil
import argparse
import numpy as np
from PIL import Image
from skimage import io
from pathlib import Path
from matplotlib import cm
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

# Pycoco
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


ROOT_DIR = os.path.abspath(".")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'pre_trained_weights/mask_rcnn_coco.h5')
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'models_new')
sys.path.append(ROOT_DIR)


# Hide deprecation warnings
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Mask-RCNN
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize


############################################################
#  Configurations
############################################################

class RoofConfig(Config):

    NAME = 'roof'
    NUM_CLASSES = 1 + 1  # background + buildings
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_RESIZE_MODE = 'square'
    
    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3

    IMAGESHAPE = np.array([IMAGE_MAX_DIM,IMAGE_MAX_DIM,3])

    # CIR
    MEAN_PIXEL = np.array([118.81442702, 94.80935892, 103.60637387])

    # IRndsm
    # MEAN_PIXEL = np.array([118.81442702,  94.80935892,  10.85432061])
    
    # CIR + ndsm
    # MEAN_PIXEL = np.array([118.81442702, 94.80935892, 103.60637387, 10.85432061])

    GPU_COUNT = 2
    IMAGES_PER_GPU = 2

    BACKBONE = 'resnet101'

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4,8, 16, 32, 64]
    

    STEPS_PER_EPOCH = 200 // IMAGES_PER_GPU
    VALIDATION_STEPS = 20

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (10, 20, 40, 80, 160)


    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5,1.0,2.0]

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256


    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 512

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = .33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 400

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 400

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.5

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.002
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.005

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
         "rpn_class_loss": 1.,
         "rpn_bbox_loss": 1,
         "mrcnn_class_loss": 1.,
         "mrcnn_bbox_loss": 1.,
         "mrcnn_mask_loss": 1.
    }

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0




class RoofInferenceConfig(RoofConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


############################################################
#  Dataset
############################################################

class RoofDataset(utils.Dataset):
    def load_roof(self, dataset_dir, subset):
        
        self.add_class('roof', 1, 'roof')
        assert subset in ['train', 'val']
        dataset_dir = os.path.join(dataset_dir, subset, 'images/')

        # Get list of images
        images = next(os.walk(dataset_dir))[2]

        # Add images
        for img in images:
            self.add_image(
                'roof',
                image_id=img,
                width=512, height=512,
                path=os.path.join(dataset_dir, img))


    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), 'masks', info['id'])

        mask = io.imread(os.path.splitext(mask_dir)[0] + '.tif')
        instances = np.unique(mask)[1:]
        all_masks = mask == instances[:, None, None]
        all_masks = np.moveaxis(all_masks, 0, -1)
        all_masks = all_masks.astype(bool)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        class_ids = np.ones([all_masks.shape[-1]], dtype=np.int32)
        return all_masks, class_ids


    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info['source'] == 'roof':
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################

def train(model):

    # Training dataset.
    dataset_train = RoofDataset()
    dataset_train.load_roof(args.dataset, 'train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = RoofDataset()
    dataset_val.load_roof(args.dataset, 'val')
    dataset_val.prepare()


    # Light augmentations
    light_augm = iaa.SomeOf((0, 4), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                    iaa.Affine(rotate=180),
                    iaa.Affine(rotate=270)]),
        iaa.Affine(
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
    ])

    # Medium augmentations
    medium_augm = iaa.SomeOf((0, 4), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Crop(percent=(0, 0.1)),
        iaa.OneOf([iaa.Affine(rotate=90),
                    iaa.Affine(rotate=180),
                    iaa.Affine(rotate=270)]),
        iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
    ])

    # Heavy augmentations
    heavy_augm = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Crop(percent=(0, 0.1)),
        iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.1))),
        iaa.Sometimes(0.5, iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)])),
        iaa.Sometimes(0.5, iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-45, 45),
        shear=(-4, 4)))
    ], random_order=True)

    shutil.copyfile('./roof_plane_segmentation.py', './models_new/latest_config.py')

    ep1 = 25
    ep2 = ep1 + 15
    ep3 = ep2 + 40
    ep4 = ep3 + 40
    ep5 = ep4 + 40

    print('Training heads')
    model.train(dataset_train, dataset_val,
                learning_rate=0.002,
                epochs=ep1,
                augmentation=light_augm,
                layers='heads')

    model.train(dataset_train, dataset_val,
                learning_rate=0.0005,
                epochs=ep2,
                augmentation=light_augm,
                layers='heads')

    print('Training resnet4+')
    model.train(dataset_train, dataset_val,
                learning_rate=0.0003,
                epochs=ep3,
                augmentation=light_augm,
                layers='4+')

    print('Training all layers')
    model.train(dataset_train, dataset_val,
                learning_rate=0.0001,
                epochs=ep4,
                augmentation=light_augm,
                layers='all')
    print('Training all layers')
    model.train(dataset_train, dataset_val,
                learning_rate=0.0001,
                epochs=ep5,
                augmentation=light_augm,
                layers='all')



############################################################
#  Predict
############################################################

def segment_region(model, data_path, output_path):

    Path(output_path + 'images/').mkdir(parents=True, exist_ok=True)
    Path(output_path + 'masks/').mkdir(parents=True, exist_ok=True)
    (_, _, file_list) = next(os.walk(data_path))
    for file in file_list:
        original_image = io.imread(data_path + file)
        results = model.detect([original_image], verbose=1)
        r = results[0]
        masks = r['masks']

        # Generate mask
        generated_mask = np.zeros((masks.shape[0], masks.shape[1]), dtype=np.uint8)
        for i in range(1, masks.shape[2]):
            generated_mask[masks[:, :, i - 1]] = i

        io.imsave(output_path + 'masks/' + os.path.splitext(file)[0] + '.tif', generated_mask)

        # Generate image with colored segments
        image_arr = np.copy(original_image)
        new_image = Image.fromarray(image_arr)
        
        cmap = cm.get_cmap('winter', masks.shape[0])

        for i in range(0, masks.shape[2]):
            image_arr[masks[:, :, i], 0] = np.clip(cmap(i)[0] + np.random.rand(1) * 255, 0, 255)
            image_arr[masks[:, :, i], 1] = np.clip(cmap(i)[1] + np.random.rand(1) * 255, 0, 255)
            image_arr[masks[:, :, i], 2] = np.clip(cmap(i)[2] + np.random.rand(1) * 255, 0, 255)

        mask_overlay = Image.fromarray(image_arr)
        new_image.putalpha(255)
        mask_overlay.putalpha(128)
        new_image = Image.alpha_composite(new_image, mask_overlay)
        new_image.save(output_path + 'images/' + os.path.splitext(file)[0] + '.tif')


############################################################
#  Evaluate
############################################################

def calculate_map(model, iou):

    dataset_val = RoofDataset()
    dataset_val.load_roof(args.dataset, 'val')
    dataset_val.prepare()
    image_ids = dataset_val.image_ids

    # VOC-Style mAP @ IoU=0.5
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, RoofInferenceConfig,
                                image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, RoofInferenceConfig), 0)

        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]

        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                            r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=iou)
        APs.append(AP)
        
    print("mAP: ", np.mean(APs))
    return np.mean(APs)
    
if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument('command',
                        metavar='<command>',
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar='/path/to/dataset/',
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar='/path/to/weights.h5',
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar='/path/to/logs/',
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar='Dataset sub-directory',
                        help='Subset of dataset to run prediction on')
    parser.add_argument('--resultout', required=False,
                        metavar='/path/to/outdir',
                        help='Where to output predicted dirs')
    args = parser.parse_args()


    # Validate arguments
    if args.command == 'train':
        assert args.dataset, "Argument --dataset is required for training"


    print('Weights: ', args.weights)
    print('Dataset: ', args.dataset)
    print('Logs: ', args.logs)

    # Configurations
    if args.command == 'train':
        config = RoofConfig()
    else:
        config = RoofInferenceConfig()
    config.display()

    # Create model
    if args.command == 'train':
        model = modellib.MaskRCNN(mode='training', config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode='inference', config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == 'coco':
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == 'last':
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == 'imagenet':
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print('Loading weights ', weights_path)
    if args.weights.lower() == 'coco':
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            'mrcnn_class_logits', 'mrcnn_bbox_fc',
            'mrcnn_bbox', 'mrcnn_mask'])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == 'train':
        train(model)
    elif args.command == 'predict':
        segment_region(model, './RoofPlaneDataset2/large_test/cir/val/images/', args.resultout.lower() + '/__TEST__/')
        segment_region(model, './RoofPlaneDataset2/no_overlap/all/cir/images/', args.resultout.lower() + '/__ALL__/')
        # segment_region(model, './RoofPlaneDataset2/large_test/cir/val/images/', './results/__TEST__/roof20220422T0706/cir/')
        # segment_region(model, './RoofPlaneDataset2/no_overlap/all/cir/images/', './results/__ALL__/roof20220422T0706/cir/')

    elif args.command == 'eval':
        print('Map: 0.5')
        calculate_map(model, 0.5)
        print('Map: 0.75')
        calculate_map(model, 0.75)