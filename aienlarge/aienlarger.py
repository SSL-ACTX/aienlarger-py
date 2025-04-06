# aienlarge.py
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
CONFIG_FILE = ".aienlarge_config.json" # Configuration file name

logging.basicConfig(level=logging.INFO) # Set default logging level to INFO
logger = logging.getLogger(__name__) # Get a logger for this module


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
    """Exception raised for unexpected API responses (non-200 OK, or unexpected JSON structure)."""
    def __init__(self, message, response=None):
        super().__init__(message)
        self.response = response
        if response is not None:
            self.status_code = response.status_code
            self.response_text = response.text
        else:
            self.status_code = None
            self.response_text = None

class ImgLargerConfigFileError(ImgLargerError):
    """Exception raised for errors related to the config file."""
    pass


class ImgLargerAPI: # Renamed to ImgLargerAPI - now async only
    """
    Asynchronous Python API wrapper for interacting with the ImgLarger PhotoAI service.

    ... (rest of class docstring as before)
    """
    def __init__(self, base_url: str = "https://photoai.imglarger.com/api/PhoAi", username: Optional[str] = None):
        """
        Initializes the asynchronous ImgLargerAPI client.

        ... (rest of constructor docstring as before)
        """
        self.base_url = base_url
        self._ensure_valid_url(base_url)

        if username: # If username is explicitly provided in constructor
            self.username = username
            self._save_config(username)
        else: # Load from config or generate
            config_username = self._load_config()
            if config_username:
                self.username = config_username
                logger.info(f"Username loaded from config: {self.username}") # Use logger
            else:
                self.username = self._generate_username()
                self._save_config(self.username)
                logger.info(f"Generated and saved new username: {self.username}") # Use logger

        # Store processing parameters internally for async version
        self._last_process_type = None
        self._last_scale_radio = None
        self._last_process_code = None
        self._retry_attempts = 3  # Number of retry attempts for network requests
        self._retry_delay_base = 1 # Base delay in seconds for exponential backoff

    def _load_config(self) -> Optional[str]:
        """Loads username from config file if it exists."""
        if not os.path.exists(CONFIG_FILE):
            return None # Config file does not exist, return None

        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config.get('username')
        except json.JSONDecodeError as e:
            logger.warning(f"Config file is corrupted or invalid JSON. Ignoring config. Error: {e}") # Log warning
            raise ImgLargerConfigFileError(f"Config file is corrupted or invalid JSON: {e}") from e # Raise specific exception
        except KeyError:
            logger.warning("Config file is missing 'username' key. Ignoring config.") # Log warning
            raise ImgLargerConfigFileError("Config file is missing 'username' key.") # Raise specific exception
        except IOError as e:
            logger.warning(f"Error reading config file. Ignoring config. Error: {e}") # Log warning
            raise ImgLargerConfigFileError(f"Error reading config file: {e}") from e # Raise specific exception
        return None # Return None if any error during loading

    def _save_config(self, username: str):
        """Saves username to config file."""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump({'username': username}, f)
            logger.debug(f"Username saved to config file: {CONFIG_FILE}") # Log debug message
        except IOError as e:
            logger.warning(f"Could not save username to config file. Error: {e}") # Log warning
            raise ImgLargerConfigFileError(f"Could not save username to config file: {e}") from e # Raise specific exception

    def _generate_username(self) -> str:
        """Generates a unique username."""
        random_prefix = str(uuid.uuid4()).split('-')[0] # Take first part of UUID for brevity
        username = f"{random_prefix}_aiimglarger"
        logger.debug(f"Generated new username: {username}") # Log debug message
        return username

    def _ensure_valid_url(self, url: str):
        """Ensures the base URL is valid."""
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise ValueError(f"Invalid base URL provided: {url}. Must be a valid URL like 'https://example.com'")

    async def _make_api_request(self, url: str, method: str, **kwargs) -> httpx.Response:
        """Handles API requests with retry logic and error handling."""
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
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                    return response # Successful response

            except httpx.RequestError as e: # Network errors (connection error, timeout, etc.)
                log_message = f"Network error during API request to {url} (attempt {attempts}/{self._retry_attempts}). Error: {e}"
                if attempts < self._retry_attempts:
                    retry_delay = self._retry_delay_base * (2**(attempts-1)) # Exponential backoff
                    logger.warning(f"{log_message} Retrying in {retry_delay:.2f} seconds...")
                    await asyncio.sleep(retry_delay) # Wait before retrying
                else:
                    logger.error(log_message)
                    raise ImgLargerAPIResponseError(f"Max retry attempts reached for API request to {url} due to network errors: {e}") from e
            except httpx.HTTPError as e: # HTTP errors (4xx, 5xx status codes)
                logger.error(f"HTTP error during API request to {url}. Status code: {e.response.status_code}, Response text: {e.response.text}")
                raise ImgLargerAPIResponseError(f"API request failed with HTTP status code: {e.response.status_code}, Response text: {e.response.text}", response=e.response) from e
            except Exception as e: # Unexpected errors
                logger.exception(f"Unexpected error during API request to {url}") # Log full exception traceback
                raise ImgLargerAPIResponseError(f"Unexpected error during API request to {url}: {e}") from e

        # Should not reach here if retry logic is correctly implemented, but for safety:
        raise ImgLargerAPIResponseError(f"API request to {url} failed after {self._retry_attempts} attempts due to unknown reasons.") # Fallback error


    async def upload_image(self, image_path: str, process_type: int, scale_radio: Optional[int] = None) -> Optional[str]:
        """
        Asynchronously uploads an image to the ImgLarger API for processing.

        ... (rest of upload_image docstring as before)
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
            scale_radio = None # Effectively ignore scale_radio for these types

        logger.info(f"Uploading image: {image_path}, process_type: {process_type}, scale_radio: {scale_radio}") # Log upload attempt

        try:
            with open(image_path, 'rb') as image_file:
                files = {'file': (os.path.basename(image_path), image_file, 'image/jpeg')}
                data = {'type': str(process_type), 'username': self.username}
                if process_type in [0, 13] and scale_radio is not None: # Only include scaleRadio if it's for type 0 or 13 and *valid*
                    data['scaleRadio'] = str(scale_radio)

                response = await self._make_api_request(upload_url, 'post', files=files, data=data) # Use _make_api_request for handling requests and retries
                json_response = response.json()

                if json_response.get("code") == 200 and json_response.get("data"):
                    process_code = json_response["data"].get("code")
                    # Store processing parameters upon successful upload (async version)
                    self._last_process_type = process_type
                    self._last_scale_radio = scale_radio # Will be None if it was ignored above
                    self._last_process_code = process_code
                    logger.info(f"Image upload successful. Process code: {process_code}") # Log success
                    return process_code
                else:
                    error_message = json_response.get("msg", "Unknown upload error")
                    logger.error(f"API upload failed. Message from API: {error_message}, API Response: {json_response}") # Log API error response
                    raise ImgLargerUploadError(f"API upload failed: {error_message}. API response: {json_response}")

        except FileNotFoundError as e:
            logger.error(f"Image file not found: {image_path}") # Log file not found
            raise
        except ValueError as e: # Re-raise ValueErrors from validation
            logger.error(f"Input validation error: {e}") # Log validation error
            raise
        except ImgLargerError as e: # Catch and re-raise custom ImgLargerErrors
            raise # Just re-raise, logging already done in _make_api_request or above
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from API during upload. Response text: {response.text if 'response' in locals() else 'No response'}")
            raise ImgLargerUploadError(f"Failed to parse JSON response from API during upload. Response text: {response.text if 'response' in locals() else 'No response'}") from e
        except Exception as e:
            logger.exception(f"Unexpected error during upload of image: {image_path}") # Log unexpected error with traceback
            raise ImgLargerUploadError(f"Unexpected error during upload: {e}") from e

    async def check_status(self, process_code: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Asynchronously checks the processing status of an image on the ImgLarger API.

        ... (rest of check_status docstring as before)
        """
        check_status_url = f"{self.base_url}/CheckStatus"

        # Retrieve stored processing parameters (async version)
        process_type = self._last_process_type
        scale_radio = self._last_scale_radio

        # Check if upload was successful and parameters are stored
        if process_type is None:
            raise ImgLargerStatusError("Processing parameters not available. Call 'upload_image' first.")


        payload = {
            "code": process_code,
            "type": process_type,
            "username": self.username,
        }
        if process_type in [0, 13] and scale_radio is not None:
            payload["scaleRadio"] = str(scale_radio)
        # No need for "elif process_type not in [0, 13] and scale_radio is not None" because validation in upload ensures scale_radio is None for these types

        logger.debug(f"Checking status for process code: {process_code}, process_type: {process_type}, scale_radio: {scale_radio}") # Log status check request

        try:
            response = await self._make_api_request(check_status_url, 'post', json=payload) # Use _make_api_request
            json_response = response.json()

            if json_response.get("code") == 200 and json_response.get("data"):
                data = json_response["data"]
                status = data.get("status")
                download_urls = data.get("downloadUrls")
                logger.info(f"Status check for process code {process_code}: Status: {status}, Download URLs available: {download_urls is not None}") # Log status check result
                return status, download_urls
            else:
                error_message = json_response.get("msg", "Unknown status check error")
                logger.error(f"API status check failed for process code {process_code}. Message from API: {error_message}, API Response: {json_response}") # Log status check error
                raise ImgLargerStatusError(f"API status check failed: {error_message}. API response: {json_response}")

        except ImgLargerError as e: # Catch and re-raise custom ImgLargerErrors
            raise # Just re-raise, logging already done in _make_api_request or above
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from API during status check. Response text: {response.text if 'response' in locals() else 'No response'}")
            raise ImgLargerStatusError(f"Failed to parse JSON response from API during status check. Response text: {response.text if 'response' in locals() else 'No response'}") from e
        except Exception as e:
            logger.exception(f"Unexpected error during status check for process code: {process_code}") # Log unexpected error with traceback
            raise ImgLargerStatusError(f"Unexpected error during status check: {e}") from e

    async def download_image(self, download_url: str, output_path_dir: str):
        """
        Asynchronously downloads the processed image from the given URL.

        ... (rest of download_image docstring as before)
        """
        logger.info(f"Downloading image from: {download_url} to directory: {output_path_dir}") # Log download attempt

        try:
            response = await self._make_api_request(download_url, 'get', follow_redirects=True) # Use _make_api_request

            # Extract filename from download URL (assuming it's the last part of the path)
            url_path = urlparse(download_url).path
            filename = os.path.basename(url_path)
            output_path = os.path.join(output_path_dir, filename) # Construct full output path

            os.makedirs(output_path_dir, exist_ok=True) # Ensure directory exists

            with open(output_path, 'wb') as output_file:
                async for chunk in response.aiter_bytes():
                    output_file.write(chunk)
            logger.info(f"Downloaded image saved to {output_path}") # Log download success

        except ImgLargerError as e: # Catch and re-raise custom ImgLargerErrors
            raise # Just re-raise, logging already done in _make_api_request or above
        except Exception as e:
            logger.exception(f"Unexpected error occurred during download from {download_url} to {output_path_dir}") # Log unexpected download error with traceback
            raise ImgLargerDownloadError(f"Unexpected error occurred during download: {e}") from e
