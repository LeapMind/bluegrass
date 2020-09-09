# Bluegrass
A sample MLOps system of LeapMind Blueoil x AWS Components (AWS IoT Greengrass, Amazon SageMaker)

## AWS Partner Network (APN) Blog
The following APN blog describes how to train models and run inference on Intel FPGA Edge Devices by using this repository.
* [Using Fewer Resources to Run Deep Learning Inference on Intel FPGA Edge Devices](https://aws.amazon.com/jp/blogs/apn/using-fewer-resources-to-run-deep-learning-inference-on-intel-fpga-edge-devices/)

## System Outline
![system outline](https://user-images.githubusercontent.com/12394960/87919648-12b48a80-cab3-11ea-819d-e799b12d9411.png)

## Setup AWS environment and devices
### Prerequisites
* AWS account
* Terasic DE10-Nano Kit x 2

### Requirements
You need following packages to setup components.
* AWS CLI

Please set up the AWS Command Line Interface (AWS CLI), check [official refferences](https://docs.aws.amazon.com/polly/latest/dg/setup-aws-cli.html).
* Ansible

You can install with `pip` command.
```shell
$ pip3 install -r requirements.txt
```

### Prepare your S3 bucket and Roles for deployment
Decide your S3 bucket name. Bucket names must be unique across all existing bucket names in Amazon S3. Please see [Bucket Restrictions and Limitations](https://docs.aws.amazon.com/AmazonS3/latest/dev/BucketRestrictions.html).
```shell
$ export BLUEGRASS_S3_BUCKET_NAME=[your S3 bucket name]
$ aws cloudformation create-stack --stack-name BluegrassS3 --capabilities CAPABILITY_NAMED_IAM --template-body file://$(pwd)/deploy/s3.yaml --parameters ParameterKey="S3BucketName",ParameterValue="${BLUEGRASS_S3_BUCKET_NAME}"
```
Please make sure your bucket is created. Check status from your [cloudformation console](https://console.aws.amazon.com/cloudformation/home).

### Blueoil x SageMaker
#### Create SageMaker Notebook for training with Blueoil
```shell
$ aws cloudformation create-stack --stack-name BlueoilSagemaker --template-body file://$(pwd)/deploy/sagemaker.yaml
```

#### Run training on SageMaker Notebook
See [blueoil_sagemaker/README.md](blueoil_sagemaker/README.md).

### Blueoil x Greengrass
#### Create certificate files for Greengrass
Run following script.
```shell
$ deploy/create_cert.sh
...
Certificate files are created in certs/xxxxxx...xxxxxx
certificateArn is "arn:aws:iot:xxxxxx:xxxxxx:cert/xxxxxx...xxxxxx"
```
Please save your `certificateArn`, this will be used when creating Greengrass components in your AWS account.

In this sample, you need to run this command twice to create certicicates for each DE10-Nano device.

#### Create Greengrass components for DE10-Nano
```shell
$ export MODEL_S3_URI=[your S3 uri]
$ export CERT_ARN1=[your certificateArn1]
$ export CERT_ARN2=[your certificateArn2]
$ aws cloudformation create-stack --stack-name Bluegrass --capabilities CAPABILITY_NAMED_IAM --template-body "$(aws cloudformation package --template-file deploy/greengrass.yaml --s3-bucket ${BLUEGRASS_S3_BUCKET_NAME})" \
--parameters \
ParameterKey="ModelS3Uri",ParameterValue="${MODEL_S3_URI}" \
ParameterKey="S3BucketName",ParameterValue="${BLUEGRASS_S3_BUCKET_NAME}" \
ParameterKey="Core01CertificateArn",ParameterValue="${CERT_ARN1}" \
ParameterKey="Core02CertificateArn",ParameterValue="${CERT_ARN2}"
```

#### Setup DE10-Nano
Download the greengrass certified SD card image from [here](http://download.terasic.com/downloads/cd-rom/de10-nano/DE10-Nano-Cloud-Native.zip)

Write that image to your SD card and boot it.
Setup network and ssh-keys to enable to access your de10nano via network without password.

Run ansible to install greengrass and setup for blueoil-inference.
```shell
$ ansible-playbook -i [IP address of Core1],[IP address of Core2] ansible/playbook.yml
```

#### Upload certificate and config files to DE10-Nano
Get your cert_id from `${CERT_ARN1}`, `${CERT_ARN2}`.
```shell
$ export CERT_ID1=$(echo ${CERT_ARN1} | sed -e 's/.*\///g')
$ export CERT_ID2=$(echo ${CERT_ARN2} | sed -e 's/.*\///g')
```
Check your thing_arn of your Core device. Find thing_arn for your core in the AWS IoT Greengrass console under [Cores](https://console.aws.amazon.com/iot/home/#/greengrass/corehub).
```shell
$ export THING_ARN1=[thing arn of your Core1]
$ export THING_ARN2=[thing arn of your Core2]
```

Set your IOT_HOST and GG_HOST.
```shell
$ export IOT_HOST=$(aws iot describe-endpoint --endpoint-type iot:Data-ATS --query 'endpointAddress' --output=text)
$ export GG_HOST=$(echo ${IOT_HOST} | sed -e 's/.*-ats/greengrass-ats/g')
```
Deploy cert and config files to DE10-Nano.
```shell
# For Device of Core1
$ ansible-playbook -i [IP address of Core1], ansible/playbook_certs_deploy.yml --extra-vars "cert_id=${CERT_ID1} thing_arn=${THING_ARN1} iot_host=${IOT_HOST} gg_host=${GG_HOST}"
# For Device of Core2
$ ansible-playbook -i [IP address of Core2], ansible/playbook_certs_deploy.yml --extra-vars "cert_id=${CERT_ID2} thing_arn=${THING_ARN2} iot_host=${IOT_HOST} gg_host=${GG_HOST}"
```

#### Associate serivce role to your account
Before first deployment, you need to associate service role to your account for greengrass deployment. This needs only once for your account.
```shell
$ ./deploy/associate_service_role.sh
{
    "AssociatedAt": "2020-XX-XXTXX:XX:XXZ"
}
```

## Initial deployment
### Manualy deployment from the AWS IoT console
After completing the above installations, you need to deploy a model and an application to devices manually.

Access to the AWS IoT console.
<div align="center"><img src="https://user-images.githubusercontent.com/12394960/88011703-fff19280-cb52-11ea-91fb-21e5a58ba841.png" width=75%></div>
Select the group and click [Action] -> [Deploy].

<div align="center"><img src="https://user-images.githubusercontent.com/12394960/88011860-62e32980-cb53-11ea-9b51-6f05336afcb5.png" width=75%></div>

The following screen is shown at the first deployment to the group.
Select [Automatic detection].

<div align="center"><img src="https://user-images.githubusercontent.com/12394960/88011874-6a0a3780-cb53-11ea-980a-7b6b0f01a89f.png" width=50%></div>

Confirm that the status is changed as [In Progress]->[Successfully Completed] as shown below.

<div align="center">
  <img src="https://user-images.githubusercontent.com/12394960/88011883-70001880-cb53-11ea-9394-a310e7afbdff.png" width=75%>
  <img src="https://user-images.githubusercontent.com/12394960/88011891-73939f80-cb53-11ea-82a7-89953f6d2215.png" width=75%>
</div>

### Check inference results
You can see the video and inference results by accessing to `http://[device's IP address]:8080`

## Update components
### Update lambda function
After updating lambda function, you can deploy it.
```shell
$ cd deploy/lambda_function
$ ./deploy_lambda.sh
```

### Update models
Please use the AWS IoT Greengrass console to change S3 path of machine learning resources.
