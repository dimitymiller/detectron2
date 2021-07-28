# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemoObjectron

import sys
module_path = os.path.abspath(os.path.join('../../Objectron/'))
if module_path not in sys.path:
    sys.path.append(module_path)
from objectron.schema import object_pb2 as object_protocol
from objectron.schema import annotation_data_pb2 as annotation_protocol
# The AR Metadata captured with each frame in the video
from objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol
from objectron.dataset import box as Box
from objectron.dataset import graphics

from os import path as osp
import json

# constants
WINDOW_NAME = "COCO detections"
COCO_Labels = [
'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--dataSave",
        type=bool,
        default=False,
        help="Do you want to save the raw detections",
    )
    parser.add_argument(
        "--visSave",
        type=bool,
        default=False,
        help="Do you want to save the detection visualisation",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Do you want to save the raw detections",
    )

    parser.add_argument(
        "--vidNum",
        type=int,
        default=None,
        help="Do you want to save the raw detections",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

lists_root_path = '../../Objectron/index'
data_root = '../data/objectron/'
def get_paths(data_root, subset, classNm = 'cup', batch = 'batch-1'):
    allNums = []
    with open(osp.join(lists_root_path, f'{classNm}_annotations_{subset}'), 'r') as f:
        for line in f:
            if f'{batch}/' not in line:
                continue
            allNums += [line.strip().replace(f'{classNm}/', '')]
    return allNums


def get_anno(ann_path):
    ann = load_annotation_sequence(ann_path)
    for item in ann:
        item[2] = 'cup'
    assert len(ann) > 0

    return ann



def get_frame_annotation(sequence, frame_id):
    """Grab an annotated frame from the sequence."""
    data = sequence.frame_annotations[frame_id]
    object_id = 0
    object_keypoints_2d = []
    object_keypoints_3d = []
    object_rotations = []
    object_translations = []
    object_scale = []
    num_keypoints_per_object = []
    object_categories = []
    annotation_types = []
    # Get the camera for the current frame. We will use the camera to bring
    # the object from the world coordinate to the current camera coordinate.
    camera = np.array(data.camera.transform).reshape(4, 4)
    for obj in sequence.objects:
        rotation = np.array(obj.rotation).reshape(3, 3)
        translation = np.array(obj.translation)
        object_scale.append(np.array(obj.scale))
        transformation = np.identity(4)
        transformation[:3, :3] = rotation
        transformation[:3, 3] = translation
        obj_cam = np.matmul(camera, transformation)
        object_translations.append(obj_cam[:3, 3])
        object_rotations.append(obj_cam[:3, :3])
        object_categories.append(obj.category)

        annotation_types.append(obj.type)

    keypoint_size_list = []
    for annotations in data.annotations:
        num_keypoints = len(annotations.keypoints)
        keypoint_size_list.append(num_keypoints)
        for keypoint_id in range(num_keypoints):
            keypoint = annotations.keypoints[keypoint_id]
            object_keypoints_2d.append(
                (keypoint.point_2d.x, keypoint.point_2d.y, keypoint.point_2d.depth))
            object_keypoints_3d.append(
                (keypoint.point_3d.x, keypoint.point_3d.y, keypoint.point_3d.z))
        num_keypoints_per_object.append(num_keypoints)
        object_id += 1

    return [object_keypoints_2d, object_keypoints_3d, object_categories, keypoint_size_list,
            annotation_types]


def load_annotation_sequence(annotation_file):
    frame_annotations = []
    with open(annotation_file, 'rb') as pb:
        sequence = annotation_protocol.Sequence()
        sequence.ParseFromString(pb.read())
        for i in range(len(sequence.frame_annotations)):
            frame_annotations.append(get_frame_annotation(sequence, i))
           # annotation, cat, num_keypoints, types
    return frame_annotations



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemoObjectron(cfg)
    
    baseFolder = '../datasets/data/objectron/videos/cup/'
    for bNum in [i for i in range(1, 6)]:
        if args.batch != None:
            if bNum != args.batch:
                continue
        bName = f'batch-{bNum}'
        folderPaths = get_paths(data_root, 'train', 'cup', bName)
        allResults = {}
        for idx, video_input in enumerate(tqdm.tqdm([baseFolder+'{}/video.MOV'.format(i) for i in folderPaths], total = len(folderPaths))):
            if args.batch != None and args.vidNum != None:
                expectedNm = f'batch-{args.batch}/{args.vidNum}'
                if expectedNm != folderPaths[idx]:
                    continue

            ## get annotation
            allResults[video_input] = {}
            print(f'Grabbing annotation for {video_input}')
            allAnnotations = get_anno(f'../datasets/data/objectron/annotations/cup/{folderPaths[idx]}.pbdata')

            video = cv2.VideoCapture(video_input)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_per_second = video.get(cv2.CAP_PROP_FPS)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
           # print(video_input)
           # exit()
            basename = os.path.basename(video_input)
            codec, file_ext = (
                ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
            )
            if codec == ".mp4v":
                warnings.warn("x264 codec not available, switching to mp4v")
            if args.output:
                if os.path.isdir(args.output):
                    output_fname = os.path.join(args.output, basename)
                    output_fname = os.path.splitext(output_fname)[0] + file_ext
                else:
                    output_fname = args.output
                assert not os.path.isfile(output_fname), output_fname
                output_file = cv2.VideoWriter(
                    filename=output_fname,
                    # some installation of opencv may not support x264 (due to its license),
                    # you can try other format (e.g. MPEG)
                    fourcc=cv2.VideoWriter_fourcc(*codec),
                    fps=float(frames_per_second),
                    frameSize=(width, height),
                    isColor=True,
                )
            assert os.path.isfile(video_input)
            #print("h1")
            bboxAnnotations = []
            for annoIdx, anno in enumerate(allAnnotations):
                annotation, annotation3D, cat, num_keypoints, types = anno
                xCoords = np.array(annotation)[1:, 0]
                yCoords = np.array(annotation)[1:, 1]
                bbox = [int(np.min(xCoords)*width), int(np.min(yCoords)*height), int(np.max(xCoords)*width), int(np.max(yCoords)*height)]
                bboxAnnotations += [bbox]


            
            #print("h2")
            allVisFrames = demo.run_on_video(video, bboxAnnotations)


            #print("h3")
            if args.visSave:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(basename, 1200,1200)
                #print('no')
            for frameIdx, (vis_frame, predictions) in enumerate(allVisFrames):
                allResults[video_input][frameIdx] = {}
                allResults[video_input][frameIdx]['scores'] = predictions['instances'].scores.cpu().tolist()
                allResults[video_input][frameIdx]['pred_classes'] =  predictions['instances'].pred_classes.cpu().tolist()
                allResults[video_input][frameIdx]['boxes'] =  predictions['instances'].pred_boxes.tensor.cpu().tolist()
                #print(predictions['features'])
                #exit()
                allResults[video_input][frameIdx]['features'] =  predictions['features'].cpu().tolist()
                bbox = bboxAnnotations[frameIdx]
                allResults[video_input][frameIdx]['gtBox'] =  bbox
            
                allCols = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
                if args.visSave:
                    strList = []
                    strColors = []
                    for bIdx, bbox in enumerate(predictions['instances'].pred_boxes.tensor.cpu().tolist()):
                        vis_frame = cv2.rectangle(vis_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), allCols[bIdx], 3)
                        clsLbl = COCO_Labels[predictions['instances'].pred_classes.cpu().tolist()[bIdx]]
                        clsScore = predictions['instances'].scores.cpu().tolist()[bIdx]
                        strList += [f'{clsLbl} : {clsScore:.2f}']
                        strColors += [allCols[bIdx]]

                    # annotation, annotation3D, cat, num_keypoints, types = allAnnotations[frameIdx]
                    # vis_frame = graphics.draw_annotation_on_image(vis_frame, annotation, num_keypoints)
                    
                    vis_frame = cv2.rectangle(vis_frame, (int(0), height-(110+(len(strList))*90)), (width, height), (125, 125, 125), -1)
                    for i, lbl in enumerate(strList[::-1]):
                        
                        vis_frame = cv2.putText(vis_frame, lbl, (int(width/4), height-((i+1)*90)), cv2.FONT_HERSHEY_SIMPLEX,
                        3, strColors[::-1][i], 3)

                              
                    cv2.imshow(basename, vis_frame)
                    fName = folderPaths[idx].replace('/', '-')
                    cv2.imwrite(f'demoFigures/{fName}/outFrame{frameIdx}.jpg', vis_frame)
                    k = cv2.waitKey(1)
                    
                    if k == 107:
                        break  # k to break
                    if k == 27:
                        exit()  # esc to quit
                # exit()
        if args.dataSave:
            with open('../results/objectron/cup/{}wFeatures.json'.format(bName), 'w') as f:
                json.dump(allResults, f)

        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()

     
