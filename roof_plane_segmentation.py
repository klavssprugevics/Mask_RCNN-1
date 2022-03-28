import os
import sys
import time
import argparse
import numpy as np
from PIL import Image
from skimage import io
from skimage.color import label2rgb
from pathlib import Path
from matplotlib import cm
import matplotlib.pyplot as plt

# Pycoco
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


ROOT_DIR = os.path.abspath(".")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'pre_trained_weights/mask_rcnn_coco.h5')
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
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
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = 'roof'

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 20 // IMAGES_PER_GPU
    # VALIDATION_STEPS = 0 // IMAGES_PER_GPU

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    # IMAGE_RESIZE_MODE = "crop"
    # IMAGE_MIN_DIM = 1024
    # IMAGE_MAX_DIM = 1024
    # IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    # POST_NMS_ROIS_TRAINING = 1000
    # POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    # USE_MINI_MASK = True
    # MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    # MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    # DETECTION_MAX_INSTANCES = 400


class RoofInferenceConfig(RoofConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = 'pad64'
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.5


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
                width=1024, height=1024,
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
    """Train the model."""
    # Training dataset.
    dataset_train = RoofDataset()
    dataset_train.load_roof(args.dataset, 'train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = RoofDataset()
    dataset_val.load_roof(args.dataset, 'val')
    dataset_val.prepare()

    # Augmentation
    # augmentation = iaa.SomeOf((0, 2), [
    #     iaa.Fliplr(0.5),
    #     iaa.Flipud(0.5),
    #     iaa.OneOf([iaa.Affine(rotate=90),
    #                iaa.Affine(rotate=180),
    #                iaa.Affine(rotate=270)]),
    #     iaa.Multiply((0.8, 1.5)),
    #     iaa.GaussianBlur(sigma=(0.0, 5.0))
    # ])


    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=3,
                # augmentation=augmentation,
                layers='heads')

    # print("Train all layers")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=40,
    #             # augmentation=augmentation,
    #             layers='all')



############################################################
#  Predict
############################################################

def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def predict_single_image(model, image_path):
    image = io.imread(image_path)
    results = model.detect([image], verbose=1)

    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                ['BG', 'roof'], r['scores'], ax=get_ax())
    plt.show()


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
    args = parser.parse_args()


    # Validate arguments
    if args.command == 'train':
        assert args.dataset, "Argument --dataset is required for training"


    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == 'train':
        config = RoofConfig()
    else:
        class InferenceConfig(RoofConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
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
        # predict_single_image(model, './data/1024_cir/val/images/pilseta2_br_0-0.jpg')
        # segment_region(model, './data/1024_cir/train/images/', './results/1024_cir/')
        pass