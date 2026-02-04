from __future__ import annotations

import json
import os
import socket
import time
import traceback
import random
from typing import TYPE_CHECKING, Dict, Tuple, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter

from glmocr.utils.logging import get_logger, get_profiler

if TYPE_CHECKING:
    from glmocr.config import OCRApiConfig

logger = get_logger(__name__)
profiler = get_profiler(__name__)


class OCRClient:
    """
    Calls a remote API for recognition without requiring local deployment of services.
    """

    def __init__(self, config: "OCRApiConfig"):
        """Initialize the OCR client.

        Args:
            config: OCRApiConfig instance.
        """
        self.config = config

        # API service configuration
        self.api_host = config.api_host
        self.api_port = config.api_port

        self.api_scheme = config.api_scheme
        if not self.api_scheme:
            self.api_scheme = "https" if self.api_port == 443 else "http"

        self.api_path = config.api_path
        if not str(self.api_path).startswith("/"):
            self.api_path = f"/{self.api_path}"

        self.api_url = config.api_url
        if not self.api_url:
            self.api_url = (
                f"{self.api_scheme}://{self.api_host}:{self.api_port}{self.api_path}"
            )

        self.api_key = config.api_key or os.getenv("GLMOCR_API_KEY")
        self.extra_headers = config.headers or {}

        # SSL verification
        self.verify_ssl = config.verify_ssl

        # If api_url is set, try to infer host/port for the connect() preflight
        try:
            parsed = urlparse(self.api_url)
            if parsed.hostname:
                self.api_host = parsed.hostname
            if parsed.port:
                self.api_port = parsed.port
        except Exception:
            pass

        # Connection timeout configuration
        self.connect_timeout = config.connect_timeout
        self.request_timeout = config.request_timeout

        # Retry configuration
        self.retry_max_attempts = getattr(config, "retry_max_attempts", 2)
        self.retry_backoff_base_seconds = getattr(
            config, "retry_backoff_base_seconds", 0.5
        )
        self.retry_backoff_max_seconds = getattr(
            config, "retry_backoff_max_seconds", 8.0
        )
        self.retry_jitter_ratio = getattr(config, "retry_jitter_ratio", 0.2)
        self.retry_status_codes = set(
            getattr(config, "retry_status_codes", [429, 500, 502, 503, 504])
        )

        # Requests session (connection pooling)
        self._session: Optional[requests.Session] = None

        # Connection pool size (from config, or default). Should be >= pipeline max_workers.
        pool_size = getattr(config, "connection_pool_size", None)
        self._pool_maxsize = pool_size if pool_size is not None else 128

        # Model information
        self.model = config.model

    def _make_session(self) -> requests.Session:
        """Create a Session with a larger connection pool for concurrent use."""
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=1,  # single API host
            pool_maxsize=self._pool_maxsize,
            max_retries=0,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def start(self):
        """Check whether the remote API service is available."""
        logger.debug("Connecting to remote API server at %s...", self.api_url)
        self.connect()
        logger.debug("Successfully connected to remote API server!")

        if self._session is None:
            self._session = self._make_session()

    def stop(self):
        """No-op: this client does not manage server lifecycle."""
        logger.debug("API recognizer does not manage server lifecycle.")

        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def _sleep_backoff(
        self, attempt_index: int, retry_after_seconds: Optional[float] = None
    ) -> None:
        if retry_after_seconds is not None and retry_after_seconds > 0:
            sleep_s = min(
                float(retry_after_seconds), float(self.retry_backoff_max_seconds)
            )
        else:
            base = float(self.retry_backoff_base_seconds)
            sleep_s = min(
                base * (2**attempt_index), float(self.retry_backoff_max_seconds)
            )

        jitter = sleep_s * float(self.retry_jitter_ratio)
        if jitter > 0:
            sleep_s = max(0.0, sleep_s + random.uniform(-jitter, jitter))

        time.sleep(sleep_s)

    @staticmethod
    def _parse_retry_after_seconds(response: requests.Response) -> Optional[float]:
        ra = response.headers.get("Retry-After")
        if not ra:
            return None
        try:
            return float(ra)
        except Exception:
            return None

    def connect(self):
        """Wait until the remote API service becomes available."""
        start_time = time.time()
        check_interval = 10

        while time.time() - start_time < self.connect_timeout:
            try:
                # Check whether the port is reachable
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(10)
                    result = sock.connect_ex((self.api_host, self.api_port))
                    if result == 0:
                        # Send a test request to the chat/completions endpoint
                        try:
                            test_payload = {
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": [{"type": "text", "text": "hello"}],
                                    }
                                ],
                                "max_tokens": 10,
                                "temperature": 0.1,
                            }
                            headers = {
                                "Content-Type": "application/json",
                                **self.extra_headers,
                            }
                            if self.api_key:
                                headers["Authorization"] = f"Bearer {self.api_key}"
                            response = requests.post(
                                self.api_url,
                                headers=headers,
                                data=json.dumps(test_payload),
                                timeout=30,
                                verify=self.verify_ssl,
                            )
                            if response.status_code == 200:
                                logger.debug(
                                    "Successfully connected to API server at %s",
                                    self.api_url,
                                )
                                if not self.model:
                                    self.model = "default"
                                return
                            else:
                                logger.warning(
                                    "API server returned status code: %s",
                                    response.status_code,
                                )
                        except requests.exceptions.RequestException as e:
                            logger.debug("Failed to connect to API: %s", e)
                        except Exception:
                            logger.debug("Unexpected error during connection test")
            except Exception:
                pass

            elapsed_time = int(time.time() - start_time)
            logger.debug(
                "Waiting for API server to be available... (%ds elapsed)",
                elapsed_time,
            )
            time.sleep(check_interval)

        raise TimeoutError(
            f"Failed to connect to API server at {self.api_url} within {self.connect_timeout} seconds"
        )

    def process(self, request_data: Dict) -> Tuple[Dict, int]:
        """
        Process a recognition request.

        Args:
            request_data: Request payload containing messages, etc.

        Returns:
            (response_dict, http_status_code)
        """
        if self._session is None:
            self._session = self._make_session()

        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Inject model if configured
        if self.model and "model" not in request_data:
            request_data["model"] = self.model

        total_attempts = int(self.retry_max_attempts) + 1
        last_error: Optional[str] = None

        for attempt in range(total_attempts):
            try:
                with profiler.measure("http_request"):
                    response = self._session.post(
                        self.api_url,
                        headers=headers,
                        data=json.dumps(request_data),
                        timeout=self.request_timeout,
                        verify=self.verify_ssl,
                    )

                if response.status_code == 200:
                    result = response.json()
                    output = result["choices"][0]["message"]["content"]
                    return {"choices": [{"message": {"content": output.strip()}}]}, 200

                status = int(response.status_code)
                body_preview = (response.text or "")[:500]

                if status in self.retry_status_codes and attempt < total_attempts - 1:
                    retry_after = self._parse_retry_after_seconds(response)
                    logger.warning(
                        "Received status %s from OCR API (attempt %d/%d). Retrying... response: %s",
                        status,
                        attempt + 1,
                        total_attempts,
                        body_preview,
                    )
                    self._sleep_backoff(
                        attempt_index=attempt, retry_after_seconds=retry_after
                    )
                    continue

                logger.warning(
                    "Received bad status code: %s, response: %s",
                    status,
                    body_preview,
                )
                return {
                    "error": "API request failed",
                    "status_code": status,
                    "response": body_preview,
                }, status

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                if attempt < total_attempts - 1:
                    logger.warning(
                        "OCR API request error (attempt %d/%d): %s. Retrying...",
                        attempt + 1,
                        total_attempts,
                        last_error,
                    )
                    self._sleep_backoff(attempt_index=attempt)
                    continue
                logger.error("Error during recognition: %s", last_error)
                logger.debug(traceback.format_exc())
                return {"error": f"Error during recognition: {last_error}"}, 500

            except Exception as e:
                logger.error("Error during recognition: %s", e)
                logger.debug(traceback.format_exc())
                return {"error": f"Error during recognition: {str(e)}"}, 500

        return {
            "error": f"API request failed after {total_attempts} attempts",
            "detail": last_error,
        }, 500
