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
from scipy.spatial.transform import Rotation as R

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
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--object",
        type=str,
        default='cup',
        help="Which object in objectron are we testing?",
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
        help="Test a specific batch?",
    )

    parser.add_argument(
        "--vidNum",
        type=int,
        default=None,
        help="Test a specific video number?",
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
data_root = '../../data/objectron/'
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
        item[2] = args.object
    assert len(ann) > 0

    return ann

def get_frame_annotation(annotation_filename):
    """Grab an annotated frame from the sequence."""
    result = []
    instances = []
    with open(annotation_filename, 'rb') as pb:
        sequence = annotation_protocol.Sequence()
        sequence.ParseFromString(pb.read())

        object_id = 0
        object_rotations = []
        object_translations = []
        object_scale = []
        num_keypoints_per_object = []
        object_categories = []
        annotation_types = []
        
        # Object instances in the world coordinate system, These are stored per sequence, 
        # To get the per-frame version, grab the transformed keypoints from each frame_annotation
        for obj in sequence.objects:
            rotation = np.array(obj.rotation).reshape(3, 3)
            translation = np.array(obj.translation)
            scale = np.array(obj.scale)
            points3d = np.array([[kp.x, kp.y, kp.z] for kp in obj.keypoints])
            instances.append((rotation, translation, scale, points3d))
        
        # Grab teh annotation results per frame
        for data in sequence.frame_annotations:
            # Get the camera for the current frame. We will use the camera to bring
            # the object from the world coordinate to the current camera coordinate.
            transform = np.array(data.camera.transform).reshape(4, 4)
            view = np.array(data.camera.view_matrix).reshape(4, 4)
            intrinsics = np.array(data.camera.intrinsics).reshape(3, 3)
            projection = np.array(data.camera.projection_matrix).reshape(4, 4)

            keypoint_size_list = []
            object_keypoints_2d = []
            object_keypoints_3d = []
            for annotations in data.annotations:
                num_keypoints = len(annotations.keypoints)
                keypoint_size_list.append(num_keypoints)
                for keypoint_id in range(num_keypoints):
                    keypoint = annotations.keypoints[keypoint_id]
                    object_keypoints_2d.append((keypoint.point_2d.x, keypoint.point_2d.y, keypoint.point_2d.depth))
                    object_keypoints_3d.append((keypoint.point_3d.x, keypoint.point_3d.y, keypoint.point_3d.z))
                num_keypoints_per_object.append(num_keypoints)
                object_id += 1
            result.append((object_keypoints_2d, object_keypoints_3d, keypoint_size_list, view, projection, intrinsics))

    return result, instances


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
    
    baseFolder = f'../../data/objectron/videos/{args.object}/'

    allBatches = {
        'bike': [i for i in range(0, 14)],
        'book': [i for i in range(1, 52)], 
        'bottle': [i for i in range(1, 47)], #bottle failed on 36
        'chair': [i for i in range(1, 47)],
        'cup': [i for i in range(1, 50)],
        'laptop': [i for i in range(0, 40)]
    }
    
    ################################################################ go through every batch for this object
    for bNum in allBatches[args.object]:
        if args.batch != None:
            if bNum != args.batch:
                continue
        bName = f'batch-{bNum}'
        folderPaths = get_paths(data_root, 'train', args.object, bName)
        allResults = {}

        ################################################################ go through every video in this batch
        for idx, video_input in enumerate(tqdm.tqdm([baseFolder+'{}/video.MOV'.format(i) for i in folderPaths], total = len(folderPaths))):
            if args.batch != None and args.vidNum != None:
                expectedNm = f'batch-{args.batch}/{args.vidNum}'
                if expectedNm != folderPaths[idx]:
                    continue
       
            
            
            
            ################################################################ get annotation and video
            allResults[video_input] = {}
            allAnnotations, allInstances = get_frame_annotation(f'../../data/objectron/annotations/{args.object}/{folderPaths[idx]}.pbdata')
            
            video = cv2.VideoCapture(video_input)
           
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_per_second = video.get(cv2.CAP_PROP_FPS)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            ############################################################## extract a rough 2D bounding boxes from annotation
            bboxAnnotations = []
            eulerAngles = []
            quatAngles = []
            
            for annoIdx, anno in enumerate(allAnnotations):
                points_2d, points_3d, num_keypoints, frame_view_matrix, frame_projection_matrix, intrinsics = allAnnotations[frame_id]
                num_instances = len(num_keypoints)

                keypoints = np.split(points_2d, np.array(np.cumsum(num_keypoints)))
                keypoints = [points.reshape(-1, 3) for points in keypoints][:-1]
                
                qA = []
                eA = []
                for instIdx in range(num_instances):
                     #object transform in the world frame
                    obj_rotation, obj_translation, obj_scale, ks = allInstances[intIdx]

                    objworld_transformation = np.identity(4)

                    objworld_transformation[:3, :3] = obj_rotation
                    objworld_transformation[:3, 3] = obj_translation

                    transformMatrix = np.matmul(frame_view_matrix, objworld_transformation)
                    rotationMatrix = transformMatrix[:3, :3]

                    r = R.from_matrix(rotationMatrix)
                    eulerAnglesA = r.as_euler('ZXY', degrees=False).tolist()
                    quaternionAnglesA = r.as_quat().tolist()
                    
                    qA += [quaternionAnglesA]
                    eA += [eulerAnglesA]
                
                eulerAngles += [eA]
                quatAngles += [qA]
                
                bboxes = []
                for k in keypoints:
                    if len(k) == 0:
                        bbox = []
                    else:
                        xCoords = np.array(k)[1:, 0]
                        yCoords = np.array(k)[1:, 1]

                        bbox = [int(np.min(xCoords)*width), int(np.min(yCoords)*height), int(np.max(xCoords)*width), int(np.max(yCoords)*height)]
                    bboxes += [bbox]
                bboxAnnotations += [bboxes]
            print(eulerAngles[:5])
            print(expectedNm)
            exit()
            ################################ run on frames and get video
            allVisFrames = demo.run_on_video(video, bboxAnnotations)



            if args.visSave:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(basename, 1200,1200)

            ################################ save results and visualise if set to true
            for frameIdx, (vis_frame, predictions) in enumerate(allVisFrames):
                
                
                allResults[video_input][frameIdx] = {}
                if len(predictions['instances']) == 0:
                    continue
                allResults[video_input][frameIdx]['scores'] = predictions['instances'].scores.cpu().tolist()
                allResults[video_input][frameIdx]['pred_classes'] =  predictions['instances'].pred_classes.cpu().tolist()
                allResults[video_input][frameIdx]['boxes'] =  predictions['instances'].pred_boxes.tensor.cpu().tolist()

                allResults[video_input][frameIdx]['predIndices'] =  predictions['predIndices'][0].cpu().tolist()
                allResults[video_input][frameIdx]['features'] = predictions['features'].cpu().tolist()
               
                bbox = bboxAnnotations[frameIdx]
                allResults[video_input][frameIdx]['gtBoxes'] =  bbox
                allResults[video_input][frameIdx]['euler'] =  eulerAngles[frameIdx]
                allResults[video_input][frameIdx]['quaternion'] =  quatAngles[frameIdx]

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


        ####################### save data
        if args.dataSave:
            with open(f'../../results/objectron/raw/{args.object}/{bName}wFeaturesTrainData.json', 'w') as f:
                json.dump(allResults, f)

        video.release()
        if args.visSave:
            cv2.destroyAllWindows()

     
