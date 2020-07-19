# Bluegrass (A sample demonstration system of LeapMind Blueoil x AWS IoT Greengrass)
## System Outline
![system outline](https://user-images.githubusercontent.com/12394960/85096104-44290480-b22e-11ea-97a9-d0e1b426edfb.png)

## Setup AWS environment and devices
### Requirements
You need following packages to setup components.
#### AWS CLI

Please set up the AWS Command Line Interface (AWS CLI), check [official User Guide](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html). You can choose the region where Amazon SageMaker and AWS IoT Greengrass are supported (see [Region Table](https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/)). 
#### Ansible

You can install with `pip` command.
```shell
$ pip3 install -r requirements.txt
```

### Prepare your Amazon S3 bucket and Roles for deployment
Decide your Amazon S3 bucket name. Bucket names must be unique across all existing bucket names in Amazon S3. Please see [Bucket Restrictions and Limitations](https://docs.aws.amazon.com/AmazonS3/latest/dev/BucketRestrictions.html).
```shell
$ export BLUEGRASS_S3_BUCKET_NAME=[your S3 bucket name]
$ aws cloudformation create-stack --stack-name BluegrassS3 --capabilities CAPABILITY_NAMED_IAM --template-body file://$(pwd)/deploy/s3.yaml --parameters ParameterKey="S3BucketName",ParameterValue="${BLUEGRASS_S3_BUCKET_NAME}"
```
Please make sure your bucket is created. Check status from your [cloudformation console](https://console.aws.amazon.com/cloudformation/home).

### Blueoil x Amazon SageMaker
#### Create an Amazon SageMaker Notebook instance for training with Blueoil
```shell
$ aws cloudformation create-stack --stack-name BlueoilSagemaker --template-body file://$(pwd)/deploy/sagemaker.yaml
```

#### Run training on Amazon SageMaker 
See [blueoil_sagemaker/README.md](blueoil_sagemaker/README.md).

### Blueoil x AWS IoT Greengrass
#### Create certificate files for AWS IoT Greengrass
Run following script.
```shell
$ deploy/create_cert.sh
...
Certificate files are created in certs/xxxxxx...xxxxxx
certificateArn is "arn:aws:iot:ap-northeast-1:123456789012:cert/xxxxxx...xxxxxx"
```
Please save your `certificateArn`, this will be used when creating Greengrass components in your AWS account.

#### Create Greengrass components for DE10-Nano
Run commands below, where `MODEL_S3_URI` will be set as a form of `s3://sagemaker-ap-northeast-1-123456789012/blueoil-sagemaker-2020-XX-XX-XX-XX-XX-XXZ/output/converted/output.tar.gz`. 
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

## Update components
### Update AWS Lambda function
After updating Lambda function, you can deploy it.
```shell
$ cd deploy/lambda_function
$ ./deploy_lambda.sh
```

### Update models
Please use the AWS IoT Greengrass console to change S3 path of machine learning resources.
