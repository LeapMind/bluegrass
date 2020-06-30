ARG base_image=blueoil/blueoil:latest
FROM ${base_image}

# For Amazon SageMaker
RUN pip install sagemaker-training
ENV PATH="/opt/ml/code:${PATH}"
ENV OUTPUT_DIR="/opt/ml/model"
ENV DATA_DIR="/"
# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM="main.py"

# # /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY script /opt/ml/code
WORKDIR /opt/ml/code
RUN chmod -R +x /opt/ml/code
ENTRYPOINT [ "" ]
