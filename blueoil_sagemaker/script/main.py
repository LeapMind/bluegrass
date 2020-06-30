#!/usr/bin/env python

# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

# A sample component that trains/converts a simple deep learning model with Blueoil.

from __future__ import print_function

import errno
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import traceback

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Execute your algorithm.
def _run(cmd):
    """Invokes your algorithm."""
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, env=os.environ)
    _, stderr = process.communicate()

    return_code = process.poll()
    if return_code:
        error_msg = 'Return Code: {}, CMD: {}, Err: {}'.format(return_code, cmd, stderr)
        raise Exception(error_msg)


def _hyperparameters_to_cmd_args(hyperparameters):
    """
    Converts our hyperparameters, in json format, into key-value pair suitable for passing to our training
    algorithm.
    """
    cmd_args_list = []

    for key, value in hyperparameters.items():
        cmd_args_list.append('--{}'.format(key))
        cmd_args_list.append(value)

    return cmd_args_list


def _extract(path):
    """Extract compressed files of input path."""
    extensions = {'.gz', '.tgz', '.zip', '.tar'}
    files = [file.path for file in os.scandir(path) if os.path.splitext(file.name)[1] in extensions]
    for file in files:
        logger.info('Extract ' + file)
        shutil.unpack_archive(file, path)

    return files


def _compress(src, dest_dir):
    base_name = os.path.basename(src)
    compressed_file = os.path.join(dest_dir, f"{base_name}.tar.gz")
    with tarfile.open(compressed_file, "w:gz") as tar:
        tar.add(src, arcname=base_name)
    return compressed_file


def _search_converted_output(model_path):
    find_output = glob.glob(os.path.join(model_path, "*/export/*/*/output"))
    if not find_output:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), os.path.join(model_path, "*/export/*/*/output")
        )
    return find_output[0]


def _train(python_executable, blueoil_cmd):
    # These are the paths to where SageMaker mounts interesting things in your container.
    prefix = '/opt/ml/'
    input_path = os.path.join(prefix, 'input/data')
    dataset_path = os.path.join(input_path, 'dataset')
    output_path = os.path.join(prefix, 'output')
    param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

    try:
        logger.info('#### Extract dataset ####')
        _extract(dataset_path)
        logger.info('#### Run training ####')
        # Amazon SageMaker makes our specified hyperparameters available within the
        # /opt/ml/input/config/hyperparameters.json.
        # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container
        with open(param_path, 'r') as tc:
            training_params = json.load(tc)
        cmd_args = _hyperparameters_to_cmd_args(training_params)
        train_cmd = [python_executable, blueoil_cmd, 'train'] + cmd_args
        logger.info(train_cmd)
        _run(train_cmd)
        logger.info('Training is completed.')
    except Exception as e:
        _error_exit(e, output_path)


def _convert(python_executable, blueoil_cmd, cmd_args):
    # These are the paths to where SageMaker mounts interesting things in your container.
    prefix = '/opt/ml/processing'
    input_path = os.path.join(prefix, 'input/data')
    dataset_path = os.path.join(input_path, 'dataset')
    output_path = os.path.join(prefix, 'output')
    model_path = os.path.join(input_path, 'model')
    # Overwrite OUTPUT_DIR
    os.environ['OUTPUT_DIR'] = model_path

    try:
        logger.info('#### Extract dataset ####')
        _extract(dataset_path)
        logger.info('#### Extract model ####')
        _extract(model_path)
        logger.info('#### Run convert ####')
        convert_cmd = [python_executable, blueoil_cmd, 'convert'] + cmd_args
        logger.info(convert_cmd)
        _run(convert_cmd)
        logger.info('#### Compress converted model ####')
        counverted_output_path = _search_converted_output(model_path)
        _compress(counverted_output_path, os.path.join(output_path, "converted"))
        logger.info('Converting is completed.')
    except Exception as e:
        _error_exit(e, output_path)


def _error_exit(error, output_path):
    # Write out an error file. This will be returned as the failureReason in the
    # DescribeTrainingJob result.
    trc = traceback.format_exc()
    with open(os.path.join(output_path, 'failure'), 'w') as s:
        s.write('Exception during training: ' + str(error) + '\n' + trc)
    # Printing this causes the exception to be in the training job logs, as well.
    logger.error('Exception during training: ' + str(error) + '\n' + trc)
    # A non-zero exit code causes the training job to be marked as Failed.
    sys.exit(255)


def main():
    # default params
    blueoil_cmd = '/home/blueoil/blueoil/cmd/main.py'
    python_executable = sys.executable

    if sys.argv[1:] and sys.argv[1] == 'convert':
        # Run convert
        cmd_args = sys.argv[2:]
        _convert(python_executable, blueoil_cmd, cmd_args)
    else:
        # Run train
        _train(python_executable, blueoil_cmd)

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)


if __name__ == '__main__':
    main()
