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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime
from io import BytesIO
import json
import logging
from multiprocessing import Pool
import os
import signal
import time

from blueoil.common import Tasks
from blueoil.utils.predict_output.output import JsonOutput
from config import (
    build_post_process,
    build_pre_process,
    load_yaml,
)
from blueoil.visualize import (
    visualize_classification,
    visualize_semantic_segmentation,
)
import cv2
import greengrasssdk
from lmnet.nnlib import NNLib
import numpy as np
from visualize import (
    draw_fps,
    visualize_object_detection_custom as visualize_od,
)


# global variable for multi process or multi thread.
nn = None
pre_process = None
post_process = None
vc = None
config = None
pool = None


class MotionJpegHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global config, pool, gg_pool
        result = None
        fps_only_network = 0.0

        pool_result = pool.apply_async(_read_camera_image, ())

        self.send_response(200)
        self.send_header(
            'Content-type',
            'multipart/x-mixed-replace; boundary=jpgboundary'
        )
        self.end_headers()

        state = 0
        displayed_waring = False
        start_time = None

        while True:
            try:
                pool_result.wait()
                window_img = pool_result.get()
                result, _, fps_only_network = _run_inference(window_img)
                pool_result = pool.apply_async(_read_camera_image, ())

                result = result[0]
                submit_flag = False
                duration = 1.0
                if config.TASK == "IMAGE.CLASSIFICATION":
                    image = visualize_classification(
                        window_img, result, config
                    )
                    now = time.time()
                    start_time = start_time or now
                    submit_flag = now - start_time > duration
                    start_time = now if submit_flag else start_time

                if config.TASK == "IMAGE.OBJECT_DETECTION":
                    prev_displayed_waring = displayed_waring
                    image, state, start_time, displayed_waring = visualize_od(
                        window_img, result, config, state, start_time, duration
                    )
                    submit_flag = (
                        displayed_waring and not prev_displayed_waring
                    )

                if config.TASK == "IMAGE.SEMANTIC_SEGMENTATION":
                    image = visualize_semantic_segmentation(
                        window_img, result, config
                    )

                draw_fps(image, fps_only_network)
                tmp = BytesIO()
                image.save(tmp, "JPEG", quality=100, subsampling=0)
                if submit_flag:
                    logging.info("Detect Warning!!!")
                    json_output = JsonOutput(
                        task=Tasks(config.TASK),
                        classes=config.CLASSES,
                        image_size=config.IMAGE_SIZE,
                        data_format=config.DATA_FORMAT,
                    )
                    json_obj = json_output(
                        np.expand_dims(result, 0), [window_img], [None]
                    )
                    gg_pool.apply_async(_submit, (json_obj, ))

                self.send_header('Content-type', 'image/jpeg')
                self.end_headers()
                self.wfile.write(tmp.getvalue())

            finally:
                self.wfile.write(b"\r\n--jpgboundary\r\n")


def _run_inference(inputs):
    global nn, pre_process, post_process
    start = time.time()

    data = pre_process(image=inputs)["image"]
    data = np.expand_dims(data, axis=0)

    network_only_start = time.time()
    result = nn.run(data)
    fps_only_network = 1.0 / (time.time() - network_only_start)

    output = post_process(outputs=result)['outputs']

    fps = 1.0 / (time.time() - start)
    return output, fps, fps_only_network


def _init_worker():
    global nn
    # ignore SIGINT in pooled process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # init
    nn.init()


def _update_exclude_score_box_threshold(config, threshold):
    if not config["POST_PROCESSOR"]:
        return config
    for d in config["POST_PROCESSOR"]:
        if "ExcludeLowScoreBox" in d:
            d["ExcludeLowScoreBox"]["threshold"] = threshold
        if "FormatCenterNet" in d:
            d["FormatCenterNet"]["score_threshold"] = threshold
    return config


def _init_camera():
    global vc
    # camera settings.
    camera_width = 320
    camera_height = 240
    camera_fps = 60
    camera_source = 0

    if hasattr(cv2, 'cv'):
        vc = cv2.VideoCapture(camera_source)
        vc.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, camera_width)
        vc.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, camera_height)
        vc.set(cv2.cv.CV_CAP_PROP_FPS, camera_fps)
    else:
        vc = cv2.VideoCapture(camera_source)
        vc.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        vc.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        vc.set(cv2.CAP_PROP_FPS, camera_fps)


def _read_camera_image():
    global vc
    _, frame = vc.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _init_gg_client():
    global gg_client
    # Creating a greengrass core sdk client
    gg_client = greengrasssdk.client('iot-data')


def _submit(json_obj):
    global gg_client
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    logging.info("Publish JSON")
    kinesis_message = {
        "request": {
            "data": json_obj,
        },
        "id": timestamp,
    }
    try:
        gg_client.publish(
            topic='kinesisfirehose/message',
            payload=json.dumps(kinesis_message),
        )
        gg_client.publish(
            topic='inference/result',
            payload=json_obj,
        )
    except Exception as e:
        logging.error("Failed to publish message: " + repr(e))


def run(model, config_file, port=80, threshold=0.5):
    global nn, pre_process, post_process, config, vc, pool, gg_client, gg_pool

    filename, file_extension = os.path.splitext(model)
    supported_files = ['.so', '.pb']

    if file_extension not in supported_files:
        raise ValueError("""
            Unknown file type. Got %s%s.
            Please check the model file (-m).
            Only .pb(protocol buffer) or .so(shared object) file is supported.
            """ % (filename, file_extension))

    if file_extension == '.so':  # Shared library
        nn = NNLib()
        nn.load(model)

    elif file_extension == '.pb':  # Protocol Buffer file
        # only load tensorflow if user wants to use GPU
        from lmnet.tensorflow_graph_runner import TensorflowGraphRunner
        nn = TensorflowGraphRunner(model)

    nn = NNLib()
    nn.load(model)
    nn.init()

    config = load_yaml(config_file)
    config = _update_exclude_score_box_threshold(config, threshold)

    pre_process = build_pre_process(config.PRE_PROCESSOR)
    post_process = build_post_process(config.POST_PROCESSOR)

    pool = Pool(processes=1, initializer=_init_camera)
    gg_pool = Pool(processes=1, initializer=_init_gg_client)

    try:
        server = HTTPServer(('', port), MotionJpegHandler)
        print("server starting")
        server.serve_forever()
    except KeyboardInterrupt:
        print("KeyboardInterrpt in server - ending server")
        vc.release()
        pool.terminate()
        pool.join()
        gg_pool.terminate()
        gg_pool.join()
        server.socket.close()
        server.shutdown()
