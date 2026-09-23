import logging
import os
import socket

import boto3
import botocore.session
from botocore.credentials import AssumeRoleCredentialFetcher, DeferredRefreshableCredentials

logger = logging.getLogger(__name__)


def role_arn(role_name: str, override_env_var: str) -> str:
    """
    ARN for an IAM role in our account. `override_env_var` (a full ARN) wins if set; otherwise the ARN is built from
    AWS_ACCOUNT, matching train_push's queue URL. Resolved lazily so importing a module doesn't require AWS env.
    """
    if override_env_var in os.environ:
        return os.environ[override_env_var]
    if "AWS_ACCOUNT" not in os.environ:
        raise RuntimeError(f"Set AWS_ACCOUNT (or {override_env_var}) to assume the {role_name} role")
    return f"arn:aws:iam::{os.environ['AWS_ACCOUNT']}:role/{role_name}"


def install_assumed_role_session(role_arn: str, region: str, session_prefix: str):
    """
    Point boto3's default session at `role_arn` with auto-refreshing credentials, so every boto3.client(...)
    call in the process transparently acts as the role. Called once at startup by scripts that touch AWS from
    non-AWS hosts (train.py on GPU cloud hosts, apply_docbot.py in the inference job).

    The source identity is whatever botocore resolves from the environment — on those hosts that's a
    *-bootstrap IAM user's long-lived keys (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY), whose only permission
    is sts:AssumeRole; the credentials the actual SQS/S3 clients receive are short-lived role-session creds.

    These credentials REFRESH: botocore re-assumes the role before each session nears expiry, so a run that
    outlasts a single STS session's 12h ceiling never loses access.
    """
    source_session = botocore.session.get_session()
    source_credentials = source_session.get_credentials()
    if source_credentials is None:
        raise RuntimeError(
            f"No AWS credentials found to assume {role_arn}. On cloud hosts set the corresponding bootstrap "
            "IAM user's AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY; to skip assumption and use ambient credentials "
            "(e.g. running locally from your root profile), set AWS_SKIP_ASSUME_ROLE=1."
        )
    fetcher = AssumeRoleCredentialFetcher(
        # pyrefly: ignore[bad-argument-type]  # create_client's wider signature satisfies the fetcher's client_creator
        client_creator=source_session.create_client,
        source_credentials=source_credentials,
        role_arn=role_arn,
        extra_args={"RoleSessionName": f"{session_prefix}-{socket.gethostname()}"[:64]},
    )
    role_session = botocore.session.get_session()
    # pyrefly: ignore[missing-attribute]  # _credentials is the canonical hook for refreshable botocore session creds
    role_session._credentials = DeferredRefreshableCredentials(
        method="assume-role", refresh_using=fetcher.fetch_credentials
    )
    boto3.setup_default_session(botocore_session=role_session, region_name=region)

    # Fail fast: force the first assume-role now, so a broken trust policy or missing source creds surfaces at
    # startup instead of hours later on the first AWS call
    identity = boto3.client("sts").get_caller_identity()
    logger.info(f"Assumed role session active as {identity['Arn']} (auto-refreshing)")
