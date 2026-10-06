"""AWS client layer for SiteSentry.

Every client works in two modes:

* ``"aws"`` — real boto3 calls via the default credential chain.
* ``"local"`` — offline fallback writing JSONL / files under
  ``data/local/<service>/``. Used when credentials are missing, no region
  is configured, boto raises, or ``SITENTRY_AWS=off``.

``make_clients()`` builds one of each. No credentials are hardcoded, and
importing this package performs no network I/O.
"""

from ._common import DEFAULT_BASE_DIR
from .cloudwatch import CloudWatchClient
from .dynamodb import DynamoClient
from .rekognition import RekognitionClient
from .s3 import S3Client
from .sns import SNSClient


def make_clients(base_dir=None) -> dict:
    """Build and return ``{"s3", "dynamodb", "sns", "rekognition",
    "cloudwatch"}`` clients. ``base_dir`` overrides the project root used
    for local-mode storage (tests pass a tmp dir)."""
    return {
        "s3": S3Client(base_dir=base_dir),
        "dynamodb": DynamoClient(base_dir=base_dir),
        "sns": SNSClient(base_dir=base_dir),
        "rekognition": RekognitionClient(base_dir=base_dir),
        "cloudwatch": CloudWatchClient(base_dir=base_dir),
    }


__all__ = [
    "DEFAULT_BASE_DIR",
    "CloudWatchClient",
    "DynamoClient",
    "RekognitionClient",
    "S3Client",
    "SNSClient",
    "make_clients",
]
