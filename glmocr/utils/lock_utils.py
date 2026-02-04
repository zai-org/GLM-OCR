"""Lock helpers for model weight conversion."""

import os
import time
import portalocker

from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


def acquire_conversion_lock(lock_file_path):
    """
    Acquire a conversion lock to ensure only one process performs conversion.

    Args:
        lock_file_path: Lock file path.

    Returns:
        lock_fd: File descriptor if acquired; otherwise None.
    """
    lock_fd = None
    try:
        # Create lock file
        lock_fd = os.open(lock_file_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)

        # Try to acquire exclusive lock (non-blocking)
        portalocker.lock(lock_fd, portalocker.LOCK_EX | portalocker.LOCK_NB)

        # Write current process info
        os.write(lock_fd, f"PID: {os.getpid()}, Time: {time.time()}".encode())
        os.fsync(lock_fd)

        logger.debug("Successfully acquired conversion lock: %s", lock_file_path)
        return lock_fd
    except (OSError, IOError, portalocker.exceptions.LockException):
        # Lock is held by another process
        if lock_fd:
            os.close(lock_fd)
        return None


def release_conversion_lock(lock_fd, lock_file_path):
    """
    Release a conversion lock.

    Args:
        lock_fd: Lock file descriptor.
        lock_file_path: Lock file path.
    """
    try:
        if lock_fd is not None:
            portalocker.unlock(lock_fd)
            os.close(lock_fd)

        # Remove lock file
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)

        logger.debug("Released conversion lock: %s", lock_file_path)
    except Exception as e:
        logger.warning("Failed to release lock %s: %s", lock_file_path, e)


def wait_for_conversion_completion(complete_path, check_interval=10):
    """
    Wait for a conversion completion flag.

    Args:
        complete_path: Completion flag file path.
        check_interval: Poll interval (seconds).

    Returns:
        bool: Whether completion was observed.
    """
    start_time = time.time()

    while True:
        # Check completion flag
        if os.path.exists(complete_path):
            return True
        logger.debug(
            "Waiting for model weight conversion to complete... (%ds elapsed)",
            int(time.time() - start_time),
        )
        time.sleep(check_interval)
