"""KFP component: send pipeline completion notifications via Slack and PagerDuty."""

from kfp import dsl


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["requests==2.31.0"],
)
def send_notification(
    model_name: str,
    pipeline_run_id: str,
    status: str,
    message: str,
    slack_webhook_url: str,
    pagerduty_routing_key: str,
    alert_on_failure: bool,
    registered_model_version: str = "",
) -> None:
    """Send Slack and optional PagerDuty notifications on pipeline completion.

    Always sends a Slack message. Triggers a PagerDuty alert only when
    status is "FAILED" and alert_on_failure is True.

    Args:
        model_name: Name of the model pipeline that completed.
        pipeline_run_id: Kubeflow pipeline run ID.
        status: Pipeline terminal status — "SUCCESS" or "FAILED".
        message: Human-readable summary message.
        slack_webhook_url: Slack incoming webhook URL (secret-injected).
        pagerduty_routing_key: PagerDuty Events API v2 routing key.
        alert_on_failure: Whether to page on failure.
        registered_model_version: Registered version number (SUCCESS only).
    """
    import logging
    import socket

    import requests

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    is_success = status.upper() == "SUCCESS"
    color = "#36a64f" if is_success else "#e01e5a"
    icon = ":white_check_mark:" if is_success else ":x:"
    hostname = socket.gethostname()

    # Build Slack Block Kit payload
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{icon} ML Pipeline {status}: {model_name}",
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Model:*\n{model_name}"},
                {"type": "mrkdwn", "text": f"*Status:*\n{status}"},
                {"type": "mrkdwn", "text": f"*Pipeline Run:*\n`{pipeline_run_id}`"},
                {"type": "mrkdwn", "text": f"*Host:*\n{hostname}"},
            ],
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Details:*\n{message}"},
        },
    ]

    if is_success and registered_model_version:
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Registered Model Version:* `{model_name} v{registered_model_version}` promoted to Staging",
                },
            }
        )

    slack_payload = {
        "attachments": [
            {
                "color": color,
                "blocks": blocks,
            }
        ]
    }

    if slack_webhook_url:
        try:
            resp = requests.post(slack_webhook_url, json=slack_payload, timeout=10)
            resp.raise_for_status()
            logger.info("Slack notification sent (status=%s)", resp.status_code)
        except requests.RequestException as exc:
            logger.warning("Failed to send Slack notification: %s", exc)
    else:
        logger.warning("No Slack webhook URL configured; skipping Slack notification")

    # PagerDuty alert on failure
    if not is_success and alert_on_failure and pagerduty_routing_key:
        pd_payload = {
            "routing_key": pagerduty_routing_key,
            "event_action": "trigger",
            "dedup_key": f"ml-pipeline-{model_name}-{pipeline_run_id}",
            "payload": {
                "summary": f"ML Pipeline FAILED: {model_name} (run {pipeline_run_id})",
                "severity": "error",
                "source": hostname,
                "component": model_name,
                "group": "ml-platform",
                "class": "pipeline-failure",
                "custom_details": {
                    "pipeline_run_id": pipeline_run_id,
                    "message": message,
                    "model_name": model_name,
                },
            },
        }
        try:
            resp = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=pd_payload,
                timeout=10,
            )
            resp.raise_for_status()
            logger.info("PagerDuty alert triggered (status=%s)", resp.status_code)
        except requests.RequestException as exc:
            logger.warning("Failed to send PagerDuty alert: %s", exc)
    elif not is_success and alert_on_failure:
        logger.warning("alert_on_failure=True but no PagerDuty routing key configured")
