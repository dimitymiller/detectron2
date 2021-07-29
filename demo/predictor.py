# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break


    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                
                yield process_predictions(frame, self.predictor(frame))

class VisualizationDemoObjectron(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def bbox_iou(self, gtBox, predBoxes, epsilon=1e-5):
        gtBoxes = gtBox.repeat(len(predBoxes), 1)
        x1 = torch.max(torch.cat((gtBoxes[:, 0].unsqueeze(1), predBoxes[:, 0].unsqueeze(1)), dim = 1), dim = 1)[0]
        y1 = torch.max(torch.cat((gtBoxes[:, 1].unsqueeze(1), predBoxes[:, 1].unsqueeze(1)), dim = 1), dim = 1)[0]
        x2 = torch.min(torch.cat((gtBoxes[:, 2].unsqueeze(1), predBoxes[:, 2].unsqueeze(1)), dim = 1), dim = 1)[0]
        y2 = torch.min(torch.cat((gtBoxes[:, 3].unsqueeze(1), predBoxes[:, 3].unsqueeze(1)), dim = 1), dim = 1)[0]

        width = (x2-x1)
        height = (y2-y1)

        width[width < 0] = 0
        height[height < 0] = 0

        area_overlap = width*height

        area_a = (gtBoxes[:, 2] - gtBoxes[:, 0]) * (gtBoxes[:, 3] - gtBoxes[:, 1])
        area_b = (predBoxes[:, 2] - predBoxes[:, 0]) * (predBoxes[:, 3] - predBoxes[:, 1])
        area_combined = area_a + area_b - area_overlap

        iou = area_overlap/ (area_combined + epsilon)

        includedArea = (area_overlap)/area_b
        return iou, includedArea

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video, gt = None):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)



        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictionsN = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictionsN)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame, predictions

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()

                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frameIdx, frame in enumerate(frame_gen):
                ###########the easy way
                #predictions = self.predictor(frame)
                

                ############the hard way
                if self.predictor.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                    frame = frame[:, :, ::-1]

                height, width = frame.shape[:2]
                image = self.predictor.aug.get_transform(frame).apply_image(frame)
                # print(image.shape)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

                inputs = [{"image": image, "height": height, "width": width}]
                # print(image)
                images = self.predictor.model.preprocess_image(inputs)
               
    
                features = self.predictor.model.backbone(images.tensor)  # set of cnn features across different scales, each scale is 256xsomething1xsomething2
               
                proposals, _ = self.predictor.model.proposal_generator(images, features, None)  # RPN, generates a set of 1000 proposal bboxes and their objectness logit

                features_ = [features[f] for f in self.predictor.model.roi_heads.box_in_features] #collects the features that are roi in features (leaves out p6?)

     
                box_features = self.predictor.model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals]) #taking in features and proposed bboxes and outputting 1000x256x7x7
                ##essentially finds the levels of the 1000 boxes
                ##then takes in those boxes and the features from that level and 
                ## uses roi align to pool into a 7x7 feature map for each of the 256 features
               
                #######
                # box_lists = [x.proposal_boxes for x in proposals]
                # x = features_
                # from typing import List 
                # from detectron2.layers import cat, nonzero_tuple
                # from detectron2.structures import Boxes            
                # def assign_boxes_to_levels(
                #     box_lists: List[Boxes],
                #     min_level: int,
                #     max_level: int,
                #     canonical_box_size: int,
                #     canonical_level: int,
                # ):
                #     """
                #     Map each box in `box_lists` to a feature map level index and return the assignment
                #     vector.

                #     Args:
                #         box_lists (list[Boxes] | list[RotatedBoxes]): A list of N Boxes or N RotatedBoxes,
                #             where N is the number of images in the batch.
                #         min_level (int): Smallest feature map level index. The input is considered index 0,
                #             the output of stage 1 is index 1, and so.
                #         max_level (int): Largest feature map level index.
                #         canonical_box_size (int): A canonical box size in pixels (sqrt(box area)).
                #         canonical_level (int): The feature map level index on which a canonically-sized box
                #             should be placed.

                #     Returns:
                #         A tensor of length M, where M is the total number of boxes aggregated over all
                #             N batch images. The memory layout corresponds to the concatenation of boxes
                #             from all images. Each element is the feature map index, as an offset from
                #             `self.min_level`, for the corresponding box (so value i means the box is at
                #             `self.min_level + i`).
                #     """
                #     box_sizes = torch.sqrt(cat([boxes.area() for boxes in box_lists]))

                #     # Eqn.(1) in FPN paper
                #     level_assignments = torch.floor(
                #         canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8)
                #     )
                #     # clamp level to (min, max), in case the box size is too large or too small
                #     # for the available feature maps
                #     level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
                #     return level_assignments.to(torch.int64) - min_level


                # def _fmt_box_list(box_tensor, batch_index: int):
                #     repeated_index = torch.full_like(
                #         box_tensor[:, :1], batch_index, dtype=box_tensor.dtype, device=box_tensor.device
                #     )
                #     return cat((repeated_index, box_tensor), dim=1)


                # def convert_boxes_to_pooler_format(box_lists: List[Boxes]):
                #     """
                #     Convert all boxes in `box_lists` to the low-level format used by ROI pooling ops
                #     (see description under Returns).

                #     Args:
                #         box_lists (list[Boxes] | list[RotatedBoxes]):
                #             A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.

                #     Returns:
                #         When input is list[Boxes]:
                #             A tensor of shape (M, 5), where M is the total number of boxes aggregated over all
                #             N batch images.
                #             The 5 columns are (batch index, x0, y0, x1, y1), where batch index
                #             is the index in [0, N) identifying which batch image the box with corners at
                #             (x0, y0, x1, y1) comes from.
                #         When input is list[RotatedBoxes]:
                #             A tensor of shape (M, 6), where M is the total number of boxes aggregated over all
                #             N batch images.
                #             The 6 columns are (batch index, x_ctr, y_ctr, width, height, angle_degrees),
                #             where batch index is the index in [0, N) identifying which batch image the
                #             rotated box (x_ctr, y_ctr, width, height, angle_degrees) comes from.
                #     """
                #     pooler_fmt_boxes = cat(
                #         [_fmt_box_list(box_list.tensor, i) for i, box_list in enumerate(box_lists)], dim=0
                #     )

                #     return pooler_fmt_boxes
                                

                # pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

                # if len(self.predictor.model.roi_heads.box_pooler.level_poolers) == 1:
                #     return self.predictor.model.roi_heads.box_pooler.level_poolers[0](x[0], pooler_fmt_boxes)

                # level_assignments = assign_boxes_to_levels(
                #     box_lists, self.predictor.model.roi_heads.box_pooler.min_level, self.predictor.model.roi_heads.box_pooler.max_level, 
                #     self.predictor.model.roi_heads.box_pooler.canonical_box_size, self.predictor.model.roi_heads.box_pooler.canonical_level
                # )

                # num_boxes = pooler_fmt_boxes.size(0)
                # num_channels = x[0].shape[1]
                # output_size = self.predictor.model.roi_heads.box_pooler.output_size[0]

                # dtype, device = x[0].dtype, x[0].device
                # output = torch.zeros(
                #     (num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device
                # )

                # for level, pooler in enumerate(self.predictor.model.roi_heads.box_pooler.level_poolers):
                    
                #     inds = nonzero_tuple(level_assignments == level)[0]
                #     pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
                #     # Use index_put_ instead of advance indexing, to avoid pytorch/issues/49852
                #     output.index_put_((inds,), pooler(x[level], pooler_fmt_boxes_level))
                #     if level == 3:
                #         print(output[inds[0],0])
                #         print(x[level][0, 0])
                        
                # exit()
                ##########################
                box_features = self.predictor.model.roi_heads.box_head(box_features)  # features of all 1k candidates
                
                predictions = self.predictor.model.roi_heads.box_predictor(box_features)
                pred_instances, pred_inds = self.predictor.model.roi_heads.box_predictor.inference(predictions, proposals)
                pred_instances = self.predictor.model.roi_heads.forward_with_given_boxes(features, pred_instances)
                # output boxes, masks, scores, etc
                predictions = self.predictor.model._postprocess(pred_instances, inputs, images.image_sizes)[0]  # scale box to orig size

                feats = box_features[pred_inds]
                predictions['features'] = []
                # predictions['allFeatures'] = box_features
                predictions['predIndices'] = pred_inds
                if gt != None and len(predictions['instances'].scores) > 0:
                    #iou greater than 0.2 and at least 80% of mass inside
                    if len(gt[frameIdx]) == 0:
                        predictions['instances'] = []
                    else:
                        ious, overlap = self.bbox_iou(torch.Tensor(gt[frameIdx]).cuda(), predictions['instances'].pred_boxes.tensor)
                        mask = ious >= 0.2
                        predictions['instances'] = predictions['instances'][mask]
                        feats = feats[mask]
                        mask = overlap[mask] >= 0.8
                        predictions['instances'] = predictions['instances'][mask]
                        feats = feats[mask]

                        predictions['features'] = feats
                    
                   
                yield frame, predictions#process_predictions(frame, predictions)

class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
