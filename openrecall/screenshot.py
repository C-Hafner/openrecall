import os
import time
from typing import List, Tuple

import mss
import numpy as np
from PIL import Image

from openrecall.config import screenshots_path, args
from openrecall.database import insert_entry, upsert_monitor, get_enabled_monitors
from openrecall.nlp import get_embedding
from openrecall.ocr import extract_text_from_image
from openrecall.utils import (
    get_active_app_name,
    get_active_window_title,
    is_user_active,
)


def mean_structured_similarity_index(
    img1: np.ndarray, img2: np.ndarray, L: int = 255
) -> float:
    """Calculates the Mean Structural Similarity Index (MSSIM) between two images.

    Args:
        img1: The first image as a NumPy array (RGB).
        img2: The second image as a NumPy array (RGB).
        L: The dynamic range of the pixel values (default is 255).

    Returns:
        The MSSIM value between the two images (float between -1 and 1).
    """
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2

    def rgb2gray(img: np.ndarray) -> np.ndarray:
        """Converts an RGB image to grayscale."""
        return 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]

    img1_gray: np.ndarray = rgb2gray(img1)
    img2_gray: np.ndarray = rgb2gray(img2)
    mu1: float = np.mean(img1_gray)
    mu2: float = np.mean(img2_gray)
    sigma1_sq = np.var(img1_gray)
    sigma2_sq = np.var(img2_gray)
    sigma12 = np.mean((img1_gray - mu1) * (img2_gray - mu2))
    ssim_index = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_index


def is_similar(
    img1: np.ndarray, img2: np.ndarray, similarity_threshold: float = 0.9
) -> bool:
    """Checks if two images are similar based on MSSIM.

    Args:
        img1: The first image as a NumPy array.
        img2: The second image as a NumPy array.
        similarity_threshold: The threshold above which images are considered similar.

    Returns:
        True if the images are similar, False otherwise.
    """
    similarity: float = mean_structured_similarity_index(img1, img2)
    return similarity >= similarity_threshold


def enumerate_and_sync_monitors() -> None:
    """
    Enumerates all connected monitors and synchronizes them with the database.

    For each detected monitor, creates or updates its entry in the monitors table
    with current resolution information. New monitors are enabled by default.
    """
    with mss.mss() as sct:
        # sct.monitors[0] is the combined view, skip it
        # sct.monitors[1:] are individual monitors
        for i in range(1, len(sct.monitors)):
            monitor = sct.monitors[i]
            width = monitor['width']
            height = monitor['height']
            name = f"Monitor {i}" if i == 1 else f"Monitor {i}"

            # Upsert the monitor (will preserve enabled status if already exists)
            upsert_monitor(
                monitor_index=i,
                name=name,
                width=width,
                height=height,
                enabled=True  # Default to enabled for new monitors
            )
    print(f"Enumerated and synced {len(sct.monitors) - 1} monitors")


def take_screenshots() -> List[Tuple[int, str, np.ndarray]]:
    """Takes screenshots of enabled monitors based on database configuration.

    Returns:
        A list of tuples: (monitor_index, monitor_name, screenshot_array)
        where screenshot_array is a NumPy array (RGB).
    """
    screenshots: List[Tuple[int, str, np.ndarray]] = []

    # Get enabled monitors from database
    enabled_monitors = get_enabled_monitors()

    # If no monitors are configured yet, fall back to all monitors or primary only
    if not enabled_monitors:
        print("No monitors configured, using default behavior")
        with mss.mss() as sct:
            monitor_indices = [1] if args.primary_monitor_only else range(1, len(sct.monitors))

            for i in monitor_indices:
                if i < len(sct.monitors):
                    monitor_info = sct.monitors[i]
                    sct_img = sct.grab(monitor_info)
                    screenshot = np.array(sct_img)[:, :, [2, 1, 0]]
                    screenshots.append((i, f"Monitor {i}", screenshot))
        return screenshots

    # Take screenshots only from enabled monitors
    with mss.mss() as sct:
        for monitor_config in enabled_monitors:
            monitor_index = monitor_config.monitor_index

            if monitor_index < len(sct.monitors):
                monitor_info = sct.monitors[monitor_index]
                sct_img = sct.grab(monitor_info)
                screenshot = np.array(sct_img)[:, :, [2, 1, 0]]
                screenshots.append((monitor_index, monitor_config.name, screenshot))
            else:
                print(f"Warning: Monitor {monitor_config.name} (index {monitor_index}) not found. Skipping.")

    return screenshots


def record_screenshots_thread() -> None:
    """
    Continuously records screenshots, processes them, and stores relevant data.

    Checks for user activity and image similarity before processing and saving
    screenshots, associated OCR text, embeddings, and active application info.
    Runs in an infinite loop, intended to be executed in a separate thread.
    """
    # TODO: Move this environment variable setting to the application's entry point.
    # HACK: Prevents a warning/error from the huggingface/tokenizers library
    # when used in environments where multiprocessing fork safety is a concern.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Enumerate and sync monitors on startup
    enumerate_and_sync_monitors()

    # Initialize last_screenshots with current screenshots
    # Store as dict: {monitor_index: (monitor_name, screenshot_array)}
    last_screenshots_dict = {}
    initial_screenshots = take_screenshots()
    for monitor_index, monitor_name, screenshot_array in initial_screenshots:
        last_screenshots_dict[monitor_index] = (monitor_name, screenshot_array)

    while True:
        if not is_user_active():
            time.sleep(3)  # Wait longer if user is inactive
            continue

        current_screenshots = take_screenshots()

        for monitor_index, monitor_name, current_screenshot in current_screenshots:
            # Get the last screenshot for this monitor if it exists
            if monitor_index not in last_screenshots_dict:
                # New monitor appeared, initialize it
                last_screenshots_dict[monitor_index] = (monitor_name, current_screenshot)
                continue

            last_monitor_name, last_screenshot = last_screenshots_dict[monitor_index]

            # Check if the screenshot has changed significantly
            if not is_similar(current_screenshot, last_screenshot):
                # Update the last screenshot for this monitor
                last_screenshots_dict[monitor_index] = (monitor_name, current_screenshot)

                # Save the screenshot
                image = Image.fromarray(current_screenshot)
                timestamp = int(time.time())
                filename = f"{timestamp}_{monitor_index}.webp"
                filepath = os.path.join(screenshots_path, filename)
                image.save(
                    filepath,
                    format="webp",
                    lossless=True,
                )

                # Extract text and create embedding
                text: str = extract_text_from_image(current_screenshot)

                # Only proceed if OCR actually extracts text
                if text.strip():
                    embedding: np.ndarray = get_embedding(text)
                    active_app_name: str = get_active_app_name() or "Unknown App"
                    active_window_title: str = get_active_window_title() or "Unknown Title"

                    # Insert entry with monitor information
                    insert_entry(
                        text=text,
                        timestamp=timestamp,
                        embedding=embedding,
                        app=active_app_name,
                        title=active_window_title,
                        monitor_id=monitor_index,
                        monitor_name=monitor_name
                    )

        time.sleep(3)  # Wait before taking the next screenshot
