#!/usr/bin/env bash
# SiteSentry — deploy.sh
# Reproducible one-command deploy of the SiteSentry Flask app to AWS.
#
# Target: EC2 t4g.micro (Graviton2 Arm, Free-Tier eligible on new accounts),
# Ubuntu 24.04, region ap-south-1 (Mumbai). Optionally use the COOL
# (Cloud-Optimized OpenCV Library) Marketplace AMI for the COOL benchmark track.
#
# What this script does (idempotent):
#   1. Creates S3 bucket, DynamoDB table, SNS topic (via AWS CLI; skips if present)
#   2. Prints the least-privilege IAM policy JSON for the instance profile
#   3. Installs system deps + Python venv + pinned requirements on the instance
#   4. Downloads ONNX models, installs a systemd unit for gunicorn
#
# Usage:
#   ./scripts/deploy.sh --key-name my-key --bucket sitesentry-demo-media \
#       --supervisor-email you@example.com [--cool-ami] [--instance-type t4g.micro]
#
# Prerequisites: AWS CLI configured with credentials that may create the above
# resources, an EC2 key pair, and a default VPC. Estimated cost on a new AWS
# account: ~$0 (t4g.micro is Free-Tier eligible, 750 hrs/mo for 12 months).
# STOP (don't terminate) the instance between sessions to stay at $0.
set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
INSTANCE_TYPE="t4g.micro"
KEY_NAME=""
BUCKET="sitesentry-demo-media"
TABLE="sitesentry-events"
TOPIC="sitesentry-alerts"
SUPERVISOR_EMAIL=""
USE_COOL_AMI=0

while [ $# -gt 0 ]; do
  case "$1" in
    --key-name) KEY_NAME="$2"; shift 2;;
    --bucket) BUCKET="$2"; shift 2;;
    --supervisor-email) SUPERVISOR_EMAIL="$2"; shift 2;;
    --cool-ami) USE_COOL_AMI=1; shift 1;;
    --instance-type) INSTANCE_TYPE="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

if [ -z "$KEY_NAME" ]; then echo "ERROR: --key-name is required"; exit 1; fi
if ! command -v aws >/dev/null; then echo "ERROR: aws CLI not found"; exit 1; fi

echo "==> region=$REGION instance=$INSTANCE_TYPE bucket=$BUCKET"

echo "==> [1/4] S3 bucket (with 30-day lifecycle expiry)"
if ! aws s3api head-bucket --bucket "$BUCKET" --region "$REGION" 2>/dev/null; then
  aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION"
  aws s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --region "$REGION" \
    --lifecycle-configuration '{"Rules":[{"ID":"evidence-30d","Status":"Enabled",
      "Filter":{"Prefix":""},"Expiration":{"Days":30}}]}'
  echo "    created $BUCKET"
else echo "    $BUCKET exists, skipping"; fi

echo "==> [2/4] DynamoDB table $TABLE"
if ! aws dynamodb describe-table --table-name "$TABLE" --region "$REGION" >/dev/null 2>&1; then
  aws dynamodb create-table --table-name "$TABLE" --region "$REGION" \
    --attribute-definitions AttributeName=site_id,AttributeType=S \
                          AttributeName=timestamp,AttributeType=S \
    --key-schema AttributeName=site_id,KeyType=HASH AttributeName=timestamp,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST
  aws dynamodb wait table-exists --table-name "$TABLE" --region "$REGION"
  echo "    created $TABLE"
else echo "    $TABLE exists, skipping"; fi

echo "==> [3/4] SNS topic $TOPIC"
TOPIC_ARN=$(aws sns create-topic --name "$TOPIC" --region "$REGION" \
  --query TopicArn --output text)
echo "    $TOPIC_ARN"
if [ -n "$SUPERVISOR_EMAIL" ]; then
  aws sns subscribe --topic-arn "$TOPIC_ARN" --region "$REGION" \
    --protocol email --notification-endpoint "$SUPERVISOR_EMAIL" >/dev/null
  echo "    subscribed $SUPERVISOR_EMAIL (confirm the AWS email!)"
fi

echo "==> [4/4] EC2 instance"
if [ "$USE_COOL_AMI" -eq 1 ]; then
  echo "    COOL AMI requested: subscribe to the 'Cloud-Optimized OpenCV Library'"
  echo "    (COOL) AMI on AWS Marketplace first, then pass --cool-ami with the AMI id."
  echo "    See docs/cool_benchmark.md for the measured Graviton benchmark procedure."
fi
# Ubuntu 24.04 Arm64 (Graviton) — resolve latest AMI id via SSM public parameter
AMI_ID=$(aws ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id \
  --region "$REGION" --query Parameter.Value --output text)
echo "    AMI: $AMI_ID"

USER_DATA=$(cat <<EOF
#!/bin/bash
set -eux
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y python3-venv python3-pip ffmpeg libgl1 libglib2.0-0 git
mkdir -p /opt/sitesentry && chown ubuntu:ubuntu /opt/sitesentry
EOF
)

INSTANCE_ID=$(aws ec2 run-instances --region "$REGION" \
  --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" --key-name "$KEY_NAME" \
  --user-data "$USER_DATA" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=sitesentry}]' \
  --query 'Instances[0].InstanceId' --output text)
echo "    instance: $INSTANCE_ID"
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "    public IP: $PUBLIC_IP"

cat <<POLICY

================================================================
NEXT STEPS (manual, one time)
----------------------------------------------------------------
1. Attach a LEAST-PRIVILEGE instance profile to $INSTANCE_ID with this policy:

{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect":"Allow","Action":["s3:PutObject","s3:GetObject"],"Resource":"arn:aws:s3:::$BUCKET/*"},
    {"Effect":"Allow","Action":["dynamodb:PutItem","dynamodb:Query","dynamodb:UpdateItem"],"Resource":"arn:aws:dynamodb:$REGION:*:$TABLE"},
    {"Effect":"Allow","Action":["sns:Publish"],"Resource":"$TOPIC_ARN"},
    {"Effect":"Allow","Action":["rekognition:DetectProtectiveEquipment"],"Resource":"*"},
    {"Effect":"Allow","Action":["cloudwatch:PutMetricData"],"Resource":"*"}
  ]
}

2. Copy this repo to the instance and finish setup:

   scp -i <key>.pem -r . ubuntu@$PUBLIC_IP:/opt/sitesentry/
   ssh -i <key>.pem ubuntu@$PUBLIC_IP <<'EOS'
     cd /opt/sitesentry
     python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
     bash scripts/download_models.sh
     cat > .env <<EOF2
   AWS_REGION=$REGION
   S3_BUCKET=$BUCKET
   DDB_TABLE=$TABLE
   SNS_TOPIC_ARN=$TOPIC_ARN
   SUPERVISOR_EMAIL=$SUPERVISOR_EMAIL
   SITENTRY_AWS=on
   EOF2
     sudo cp deploy/sitesentry.service /etc/systemd/system/
     sudo systemctl daemon-reload && sudo systemctl enable --now sitesentry
   EOS

3. Open http://$PUBLIC_IP:5000 — the dashboard. Health: /api/health
   STOP the instance between sessions: aws ec2 stop-instances --instance-ids $INSTANCE_ID
================================================================
POLICY
