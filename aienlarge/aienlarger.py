import httpx
import json
import os
import time
from urllib.parse import urlparse
from typing import Optional, Tuple, List, Dict
import uuid
import logging
import asyncio

USER_AGENT_HEADER = {"User-Agent": "Dart/3.5 (dart:io)"}
CONFIG_FILE = ".aienlarge_config.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImgLargerError(Exception):
    """Base class for all ImgLarger API errors."""
    pass


class ImgLargerUploadError(ImgLargerError):
    """Exception raised when image upload fails."""
    pass


class ImgLargerStatusError(ImgLargerError):
    """Exception raised when status check fails."""
    pass


class ImgLargerDownloadError(ImgLargerError):
    """Exception raised when image download fails."""
    pass


class ImgLargerInvalidProcessTypeError(ImgLargerError):
    """Exception raised for invalid process type."""
    pass


class ImgLargerInvalidScaleRadioError(ImgLargerError):
    """Exception raised for invalid scale radio value."""
    pass


class ImgLargerAPIResponseError(ImgLargerError):
    """Exception raised for unexpected API responses."""

    def __init__(self, message, response=None):
        super().__init__(message)
        self.response = response
        self.status_code = response.status_code if response else None
        self.response_text = response.text if response else None


class ImgLargerConfigFileError(ImgLargerError):
    """Exception raised for errors related to the config file."""
    pass


class ImgLargerAPI:
    """Asynchronous Python API wrapper for interacting with the ImgLarger PhotoAI service."""

    def __init__(self, base_url: str = "https://photoai.imglarger.com/api/PhoAi", username: Optional[str] = None):
        """
        Initializes the asynchronous ImgLargerAPI client.
        A username is either provided, loaded from config, or generated.
        """
        self.base_url = base_url
        self._ensure_valid_url(base_url)

        if username:
            self.username = username
            self._save_config(username)
        else:
            config_username = self._load_config()
            if config_username:
                self.username = config_username
                logger.info(f"Username loaded from config: {self.username}")
            else:
                self.username = self._generate_username()
                self._save_config(self.username)
                logger.info(f"Generated and saved new username: {self.username}")

        # Store processing parameters internally to preserve state across async calls.
        self._last_process_type = None
        self._last_scale_radio = None
        self._last_process_code = None

        # Retry configuration for robust API communication.
        self._retry_attempts = 3
        self._retry_delay_base = 1

    def _load_config(self) -> Optional[str]:
        """Loads username from config file if it exists."""
        if not os.path.exists(CONFIG_FILE):
            return None

        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config.get('username')
        except json.JSONDecodeError as e:
            logger.warning(f"Config file is corrupted. Ignoring config. Error: {e}")
            raise ImgLargerConfigFileError(f"Config file is corrupted: {e}") from e
        except KeyError:
            logger.warning("Config file is missing 'username' key. Ignoring config.")
            raise ImgLargerConfigFileError("Config file is missing 'username' key.")
        except IOError as e:
            logger.warning(f"Error reading config file. Ignoring config. Error: {e}")
            raise ImgLargerConfigFileError(f"Error reading config file: {e}") from e

        return None

    def _save_config(self, username: str):
        """Saves username to config file."""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump({'username': username}, f)
            logger.debug(f"Username saved to config file: {CONFIG_FILE}")
        except IOError as e:
            logger.warning(f"Could not save username to config file. Error: {e}")
            raise ImgLargerConfigFileError(f"Could not save username to config file: {e}") from e

    def _generate_username(self) -> str:
        """Generates a unique username using UUID."""
        random_prefix = str(uuid.uuid4()).split('-')[0]
        username = f"{random_prefix}_aiimglarger"
        logger.debug(f"Generated new username: {username}")
        return username

    def _ensure_valid_url(self, url: str):
        """Validates the base URL format."""
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise ValueError(f"Invalid base URL provided: {url}.")

    async def _make_api_request(self, url: str, method: str, **kwargs) -> httpx.Response:
        """
        Handles API requests with retry logic and centralized error handling.
        Implements exponential backoff for transient network issues.
        """
        attempts = 0
        while attempts < self._retry_attempts:
            attempts += 1
            try:
                async with httpx.AsyncClient(headers=USER_AGENT_HEADER, timeout=30) as client:
                    if method == 'post':
                        response = await client.post(url, **kwargs)
                    elif method == 'get':
                        response = await client.get(url, **kwargs)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")

                    response.raise_for_status()
                    return response

            except httpx.RequestError as e:  # Handle network-related errors
                log_message = f"Network error during API request to {url} (attempt {attempts}/{self._retry_attempts}). Error: {e}"
                if attempts < self._retry_attempts:
                    retry_delay = self._retry_delay_base * (2 ** (attempts - 1))
                    logger.warning(f"{log_message} Retrying in {retry_delay:.2f} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(log_message)
                    raise ImgLargerAPIResponseError(f"Max retry attempts reached for API request to {url} due to network errors: {e}") from e

            except httpx.HTTPError as e:  # Handle HTTP status code errors
                logger.error(f"HTTP error during API request to {url}. Status code: {e.response.status_code}, Response text: {e.response.text}")
                raise ImgLargerAPIResponseError(f"API request failed with HTTP status code: {e.response.status_code}, Response text: {e.response.text}",
                                                 response=e.response) from e

            except Exception as e:  # Handle unexpected errors
                logger.exception(f"Unexpected error during API request to {url}")
                raise ImgLargerAPIResponseError(f"Unexpected error during API request to {url}: {e}") from e

        raise ImgLargerAPIResponseError(f"API request to {url} failed after {self._retry_attempts} attempts.")

    async def upload_image(self, image_path: str, process_type: int, scale_radio: Optional[int] = None) -> Optional[str]:
        """
        Asynchronously uploads an image to the ImgLarger API for processing.

        Args:
            image_path: Path to the image file.
            process_type:  Type of processing to apply (0, 1, 2, 3, 13).
            scale_radio:  Scaling factor (2, 4, 8) for process_type 0 or 13.

        Returns:
            The process code if the upload is successful, otherwise None.

        Raises:
            ValueError: If image_path is invalid.
            FileNotFoundError: If the image file is not found.
            ImgLargerInvalidProcessTypeError: If process_type is invalid.
            ImgLargerInvalidScaleRadioError: If scale_radio is invalid for the given process_type.
            ImgLargerUploadError: If the API upload fails.
        """
        upload_url = f"{self.base_url}/Upload"

        # Input Validation
        if not isinstance(image_path, str) or not image_path:
            raise ValueError("Image path must be a non-empty string.")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not isinstance(process_type, int) or process_type not in [0, 1, 2, 3, 13]:
            raise ImgLargerInvalidProcessTypeError(f"Invalid process_type: {process_type}. Must be one of 0, 1, 2, 3, or 13.")
        if process_type in [0, 13]:
            if scale_radio is not None and scale_radio not in [2, 4, 8]:
                raise ImgLargerInvalidScaleRadioError(f"Invalid scaleRadio value for process_type {process_type}. Valid values are 2, 4, or 8.")
        elif process_type in [1, 2, 3] and scale_radio is not None:
            logger.warning(f"scaleRadio is not supported for process_type {process_type} and will be ignored.")
            scale_radio = None  # Effectively ignore scale_radio for these types

        logger.info(f"Uploading image: {image_path}, process_type: {process_type}, scale_radio: {scale_radio}")

        try:
            with open(image_path, 'rb') as image_file:
                files = {'file': (os.path.basename(image_path), image_file, 'image/jpeg')}
                data = {'type': str(process_type), 'username': self.username}

                # Conditionally include scaleRadio based on process_type and its validity
                if process_type in [0, 13] and scale_radio is not None:
                    data['scaleRadio'] = str(scale_radio)

                response = await self._make_api_request(upload_url, 'post', files=files, data=data)
                json_response = response.json()

                if json_response.get("code") == 200 and json_response.get("data"):
                    process_code = json_response["data"].get("code")
                    # Store processing parameters for subsequent calls.
                    self._last_process_type = process_type
                    self._last_scale_radio = scale_radio
                    self._last_process_code = process_code

                    logger.info(f"Image upload successful. Process code: {process_code}")
                    return process_code
                else:
                    error_message = json_response.get("msg", "Unknown upload error")
                    logger.error(f"API upload failed. Message from API: {error_message}, API Response: {json_response}")
                    raise ImgLargerUploadError(f"API upload failed: {error_message}. API response: {json_response}")

        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            raise
        except ValueError:
            logger.error(f"Input validation error.")
            raise
        except ImgLargerError:
            raise  # Re-raise, logging already done in _make_api_request or above
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from API during upload. Response text: {response.text if 'response' in locals() else 'No response'}")
            raise ImgLargerUploadError(f"Failed to parse JSON response from API during upload. Response text: {response.text if 'response' in locals() else 'No response'}") from e
        except Exception as e:
            logger.exception(f"Unexpected error during upload of image: {image_path}")
            raise ImgLargerUploadError(f"Unexpected error during upload: {e}") from e

    async def check_status(self, process_code: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Asynchronously checks the processing status of an image on the ImgLarger API.

        Args:
            process_code: The process code returned by the upload_image method.

        Returns:
            A tuple containing the status and a list of download URLs (if available).

        Raises:
            ImgLargerStatusError: If the API status check fails.
        """
        check_status_url = f"{self.base_url}/CheckStatus"

        # Retrieve stored processing parameters.
        process_type = self._last_process_type
        scale_radio = self._last_scale_radio

        if process_type is None:
            raise ImgLargerStatusError("Processing parameters not available. Call 'upload_image' first.")

        payload = {
            "code": process_code,
            "type": process_type,
            "username": self.username,
        }

        # Conditionally include scaleRadio in the payload.
        if process_type in [0, 13] and scale_radio is not None:
            payload["scaleRadio"] = str(scale_radio)

        logger.debug(f"Checking status for process code: {process_code}, process_type: {process_type}, scale_radio: {scale_radio}")

        try:
            response = await self._make_api_request(check_status_url, 'post', json=payload)
            json_response = response.json()

            if json_response.get("code") == 200 and json_response.get("data"):
                data = json_response["data"]
                status = data.get("status")
                download_urls = data.get("downloadUrls")
                logger.info(f"Status check for process code {process_code}: Status: {status}, Download URLs available: {download_urls is not None}")
                return status, download_urls
            else:
                error_message = json_response.get("msg", "Unknown status check error")
                logger.error(f"API status check failed for process code {process_code}. Message from API: {error_message}, API Response: {json_response}")
                raise ImgLargerStatusError(f"API status check failed: {error_message}. API response: {json_response}")

        except ImgLargerError:
            raise  # Re-raise, logging already done in _make_api_request or above
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from API during status check. Response text: {response.text if 'response' in locals() else 'No response'}")
            raise ImgLargerStatusError(f"Failed to parse JSON response from API during status check. Response text: {response.text if 'response' in locals() else 'No response'}") from e
        except Exception as e:
            logger.exception(f"Unexpected error during status check for process code: {process_code}")
            raise ImgLargerStatusError(f"Unexpected error during status check: {e}") from e

    async def download_image(self, download_url: str, output_path_dir: str):
        """
        Asynchronously downloads the processed image from the given URL.

        Args:
            download_url: The URL of the processed image to download.
            output_path_dir: The directory to save the downloaded image to.

        Raises:
            ImgLargerDownloadError: If an error occurs during the download.
        """
        logger.info(f"Downloading image from: {download_url} to directory: {output_path_dir}")

        try:
            response = await self._make_api_request(download_url, 'get', follow_redirects=True)

            # Extract filename from download URL.
            url_path = urlparse(download_url).path
            filename = os.path.basename(url_path)
            output_path = os.path.join(output_path_dir, filename)

            os.makedirs(output_path_dir, exist_ok=True)

            with open(output_path, 'wb') as output_file:
                async for chunk in response.aiter_bytes():
                    output_file.write(chunk)

            logger.info(f"Downloaded image saved to {output_path}")

        except ImgLargerError:
            raise  # Re-raise, logging already done in _make_api_request or above
        except Exception as e:
            logger.exception(f"Unexpected error occurred during download from {download_url} to {output_path_dir}")
            raise ImgLargerDownloadError(f"Unexpected error occurred during download: {e}") from e