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

# Create certificate
aws iot create-keys-and-certificate --set-as-active --certificate-pem-outfile "cert.pem" --public-key-outfile "public.key" --private-key-outfile "private.key" | tee cert_result.log > /dev/null

# Move files to certs directory
CERT_ID=$(grep certificateArn cert_result.log | awk -F cert/ '{print $2}' | sed -e 's/",//g')
CERT_ARN=$(grep certificateArn cert_result.log | awk '{print $2}' | sed -e 's/,//g')
mkdir -p certs/${CERT_ID}
mv cert.pem public.key private.key cert_result.log certs/${CERT_ID}/

echo ""
echo "Certificate files are created in certs/${CERT_ID}"
echo "certificateArn is ${CERT_ARN}"
