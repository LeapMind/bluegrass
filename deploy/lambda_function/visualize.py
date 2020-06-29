# -*- coding: utf-8 -*-
# Copyright 2019 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Inference results visualization (decoration) functions and helpers."""
from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import time

import cv2
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

FONT = "DejaVuSans.ttf"
STATE_NORMAL = "NORMAL"
STATE_WARNING = "WARNING"
STATE_CLEAR = "CLEAR"


def _mask_image(image):
    # Mask right-half image with Red color
    height, width = image.shape[:2]
    image_chunk = image[:height, width // 2:]
    mask_rect = np.ones(image_chunk.shape, dtype=np.uint8)
    mask_rect[:, :, 0] = np.ones(image_chunk.shape[:2], dtype=np.uint8) * 255
    image[:height, width // 2:width] = cv2.addWeighted(
        image_chunk, 0.5, mask_rect, 0.5, 1.0
    )
    return image


def _scale_boxes(post_processed, input_image_shape, predict_image_shape):
    height_scale = input_image_shape[0] / float(predict_image_shape[0])
    width_scale = input_image_shape[1] / float(predict_image_shape[1])

    predict_boxes = np.copy(post_processed)
    predict_boxes[:, 0] *= width_scale
    predict_boxes[:, 1] *= height_scale
    predict_boxes[:, 2] *= width_scale
    predict_boxes[:, 3] *= height_scale

    return predict_boxes


def _decide_class(preds, tuning_threshold=0.0):
    class_id = 0
    score = 0.0
    for pred in preds:
        current_score = pred[1]
        if current_score - tuning_threshold >= score:
            class_id = int(pred[0])
            score = current_score
    return class_id, score


def _gather_prediction(predict_boxes):
    box_preds_list = []
    indices = {}
    for predict_box in predict_boxes:
        box = predict_box[:4].tolist()
        pred = predict_box[4:]
        key = "{}-{}-{}-{}".format(*box)
        if key in indices:
            box_preds_list[indices[key]]["preds"].append(pred)
        else:
            indices[key] = len(box_preds_list)
            box_preds_list.append({
                "box": box,
                "preds": [pred],
            })

    uniq_boxes = []
    for box_preds in box_preds_list:
        class_id, score = _decide_class(box_preds["preds"])
        box = box_preds["box"]
        uniq_boxes.append({
            "box": box,
            "class_id": class_id,
            "score": score,
        })

    return uniq_boxes


def _get_state(box, center_width):
    # Check if the center of box is not in restricted area
    mid = box["box"][0] + (box["box"][2] / 2)
    return (mid < center_width), box["class_id"]


def _get_total_state(states, ng_class_id):
    box_in_restricted_area = False
    for state, class_id in states:
        if not state:
            if class_id == ng_class_id:
                return STATE_WARNING
            box_in_restricted_area = True
    return STATE_CLEAR if box_in_restricted_area else STATE_NORMAL


def visualize_object_detection_custom(
        image, post_processed, config, prev_state, start_time, duration):
    """Draw object detection result boxes to image.

    Args:
        image (np.ndarray): A inference input RGB image to be draw.
        post_processed (np.ndarray): A one batch output of model be
            already applied post process. Format is defined at
            https://github.com/blue-oil/blueoil/blob/master/docs/specification/output_data.md
        config (EasyDict): Inference config.
        prev_state (string): A previous state, "NORMAL" or "WARNING" or "CLEAR"
        start_time (float): UNIX time when state was changed to current state
        duration (float): Duration(sec) for waiting to change status displayed

    Returns:
        PIL.Image.Image: drawn image object.
        String: A current state ("NORMAL" or "WARNING" or "CLEAR")
        Float: UNIX time when state was changed to current state
        Bool: A flag of which "WARNING" is displayed or not

    """

    colorWarning = (255, 0, 0)
    colorClear = (0, 255, 0)
    box_font = PIL.ImageFont.truetype(FONT, 10)
    state_font = PIL.ImageFont.truetype(FONT, 20)

    classes = config.CLASSES
    ng_class_id = classes.index("face") if "face" in classes else 0
    start_time = start_time or time.time()

    center_width = image.shape[1] // 2
    predict_boxes = _scale_boxes(
        post_processed, image.shape, config.IMAGE_SIZE
    )

    # Gather and remove duplicate box in different classes
    uniq_boxes = _gather_prediction(predict_boxes)
    states = [_get_state(box, center_width) for box in uniq_boxes]
    total_state = _get_total_state(states, ng_class_id)

    image = PIL.Image.fromarray(_mask_image(image))
    draw = PIL.ImageDraw.Draw(image)
    for uniq_box, state in zip(uniq_boxes, states):
        box = uniq_box["box"]
        class_id = uniq_box["class_id"]
        xy = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        color = colorWarning if class_id == ng_class_id else colorClear
        prefix = "[OK]" if state[0] or (state[1] != ng_class_id) else "[NG]"
        txt = "{:s} {:s}: {:.3f}".format(
            prefix, classes[class_id], float(uniq_box["score"])
        )
        draw.rectangle(xy, outline=color)
        draw.text([box[0], box[1]], txt, fill=color, font=box_font)

    if prev_state != total_state:
        start_time = time.time()

    elapsed_time = float(time.time() - start_time)
    right_corner = [center_width + 60, 0]

    displayed_waring = False
    if total_state == STATE_WARNING and elapsed_time >= duration:
        draw.text(right_corner, "WARNING", fill=colorWarning, font=state_font)
        displayed_waring = True
    elif total_state == STATE_CLEAR and elapsed_time >= duration:
        draw.text(right_corner, "  CLEAR", fill=colorClear, font=state_font)

    return image, total_state, start_time, displayed_waring


def draw_fps(pil_image, fps_only_network):
    """Draw FPS information to image object.

    Args:
        pil_image (PIL.Image.Image): Image object to be draw FPS.
        fps_only_network (float): FPS of network only (not pre/post process).

    Returns:

    """
    font_size = 14
    PIL.ImageDraw.Draw(pil_image).text(
        (10, pil_image.height - font_size - 5),
        "FPS: {:.1f}".format(fps_only_network),
        fill=(0, 0, 255),
        font=PIL.ImageFont.truetype(FONT, font_size),
    )
