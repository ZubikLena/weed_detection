"""
Main script for the detection pipeline.
"""
import cv2
import glob
from collections import OrderedDict
from camera_factory import CameraFactory
from image_processing_factory import ImageProcessingFactory
from classification_factory import ClassificationFactory


import communication as comm

######## CNN ########
# Ordered dictionary describing the processing steps in the correct order with their settings
processing_config = OrderedDict(
    {
        "green_amplifiy": {},
        "blur": {
            "kernel_size": 3
        },
        "adaptive_threshold": {},
        "dilate": {
            "kernel_size": 5,
            "iterations": 7,
        },
        "erode": {
            "kernel_size": 3,
            "iterations": 1,
        }
    }
)
# Example of how to use the processing factory
segmentation = ImageProcessingFactory.create_image_processing(
    "plant_segmentation", processing_config)

# Example of how to use the camera factory
camera = CameraFactory.create_camera(
    "filecam", {"folder": "dataset", "width": 640, "height": 480})

classifier = ClassificationFactory.create_classifier(
    "cnn", {"model_path": "model.h5", "min_leaf_area": 1300, "cluster_distance_threshold": 10})


######## Contour ########
# processing_config = OrderedDict(
#     {
#         "green_amplifiy": {},
#         "blur": {
#             "kernel_size": 3
#         },
#         "adaptive_threshold": {},
#         "dilate": {
#             "kernel_size": 5,
#             "iterations": 7,
#         },
#         "erode": {
#             "kernel_size": 3,
#             "iterations": 1,
#         }
#     }
# )
# # Example of how to use the processing factory
# segmentation = ImageProcessingFactory.create_image_processing(
#     "plant_segmentation", processing_config)

# # Example of how to use the camera factory
# camera = CameraFactory.create_camera(
#     "filecam", {"folder": "dataset"})

# classifier = ClassificationFactory.create_classifier(
#     "leaf_contour", {"min_leaf_area": 1300, "cluster_distance_threshold": 10,"max_carrot_cnts": 30, "max_weed_cnts": 30})

# def create_contour_dataset(segmentation, classifier):
#     carrot_folder = "data/train/carrot"
#     weed_folder = "data/train/weed"
#     carrot_images = glob.glob(carrot_folder + "/*.png")
#     weed_images = glob.glob(weed_folder + "/*.png")

#     carrot_images = [cv2.imread(image) for image in carrot_images]
#     weed_images = [cv2.imread(image) for image in weed_images]

#     carrot_masks = [segmentation.process_image(image) for image in carrot_images]
#     weed_masks = [segmentation.process_image(image) for image in weed_images]

#     # Release carrot_images
#     carrot_images = None
#     weed_images = None

#     classifier.create_contour_dataset(carrot_masks, weed_masks)

# create_contour_dataset(segmentation, classifier)


image = camera.get_image()
# Image writer
writer = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', writer, 20.0, (image.shape[1], image.shape[0]))

comm.wait_for_start()
while(True):
    if comm.check_for_stop():
        comm.wait_for_start()
        
    # Get the image from the camera (in BGR format)
    t0 = time.time()
    image = camera.get_image()

    if image is None:
        print("No image received from camera")
        break

    # Process the image
    segmentation_mask = segmentation.process_image(image)

    # Classify the plants
    classification_result = classifier.classify_plants(
        image, segmentation_mask)
    # Communication
    for plant_label, plant_bbox in zip(classification_result["labels"], classification_result["bboxes"]):
        bbox_center = (plant_bbox[0] + plant_bbox[2] / 2, plant_bbox[1] + plant_bbox[3] / 2)
        comm.send_data_to_arduino(bbox_center, plant_label)

    # Show the image and classification result
    # cv2.imshow("Image", image)
    # cv2.imshow("Segmentation", segmentation_mask)
    for label, bbox in zip(classification_result["labels"], classification_result["bboxes"]):
        # Draw the bounding box
        if label == 0:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] +
                        bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] +
                        bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)
        # # Draw the label
        # cv2.putText(image, str(
        #     label), (bbox[0]+5, bbox[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.imshow("Classification", image)
    # Write image to video file
    out.write(image)
    # cv2.waitKey(10)
out.release()