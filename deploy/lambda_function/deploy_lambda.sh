#!/bin/bash -eu
# Copyright (c) 2020 LeapMind Inc. All Rights Reserved.
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

FUNCTION_NAME=${1-bluegrass_inference_server}

# Update Function Code
echo ""
echo "### Updating function code: ${FUNCTION_NAME}"
zip ${FUNCTION_NAME}.zip *.py
aws lambda update-function-code --function-name ${FUNCTION_NAME} --zip-file fileb://${FUNCTION_NAME}.zip
rm ${FUNCTION_NAME}.zip

# Publish Version
echo ""
echo "### Publishing version"
aws lambda publish-version --function-name ${FUNCTION_NAME}

# Update Alias
LATEST_VERSION=$(aws lambda list-versions-by-function --function-name ${FUNCTION_NAME} --query 'Versions[-1].Version' --output=text | tail -1)
ALIAS_NAME=${2-GGAlias}
echo ""
echo "### Update alias: ${ALIAS_NAME}"
aws lambda update-alias --function-name ${FUNCTION_NAME} --name ${ALIAS_NAME} --function-version ${LATEST_VERSION}

# Create deployment
for GG_GROUP_ID in $(aws greengrass list-groups --query 'Groups[].Id' --output=text)
do
    GG_LATEST_VERSION_ID=$(aws greengrass list-group-versions --group-id ${GG_GROUP_ID} --query 'Versions[0].Version' --output=text --no-paginate)
    echo ""
    echo "### Creating deployment: ${GG_GROUP_ID}"
    aws greengrass create-deployment --deployment-type NewDeployment --group-id ${GG_GROUP_ID} --group-version-id ${GG_LATEST_VERSION_ID}
done
