# aienlarge-py

**An Asynchronous Python API Wrapper for AI Enlarger**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python Versions](https://img.shields.io/pypi/pyversions/aienlarge-py.svg?logo=python&logoColor=white)](https://pypi.org/project/aienlarge-py/)

`aienlarge-py` is an **unofficial** Python library that provides an asynchronous interface to the AI Enlarger image enhancement service. It allows you to programmatically upload images, enhance or enlarge them using AI-powered processes, monitor processing status, and download enhanced results. It leverages `httpx` for efficient non-blocking I/O via `asyncio`.

---

## 🚀 Features

- **Asynchronous API:** Built with `httpx` and `asyncio` for efficient concurrency.
- **Image Upload:** Send images for processing with specified enhancement types and scaling.
- **Status Monitoring:** Check the status of your image processing tasks.
- **Image Downloading:** Retrieve processed images when ready.
- **Robust Error Handling:** Custom exceptions for granular error management.
- **Automatic Configuration:** Generates and persists a username locally for API usage.
- **Retry Logic:** Auto-retries failed requests with exponential backoff.
- **Integrated Logging:** Uses Python’s `logging` module for traceability.

---

## ⚠️ Disclaimer

This library is **unofficial** and not affiliated with or endorsed by AI Enlarger or its developers. Use at your own discretion and comply with any terms of service or usage limits set by AI Enlarger.

---

## 📦 Installation

Install via GitHub:

```bash
pip install git+https://github.com/SSL-ACTX/aienlarger-py.git#egg=aienlarge-py
```

Or install from PyPI:

```bash
pip install aienlarge-py
```

---

## 🧪 Usage Example

```python
import asyncio
import aienlarge
import os

async def main():
    api = aienlarge.ImgLargerAPI()

    image_path = "path/to/your/image.jpg"
    output_dir = "output_images"

    os.makedirs(output_dir, exist_ok=True)

    try:
        # Upload image for 2x enhancement using Default Image Upscaler
        process_code = await api.upload_image(image_path, process_type=0, scale_radio=2)

        if process_code:
            print(f"Upload successful. Process code: {process_code}")

            while True:
                status, download_urls = await api.check_status(process_code)
                print(f"Status: {status}")

                if status == "done" and download_urls:
                    print("Processing complete. Downloading images...")
                    for i, url in enumerate(download_urls):
                        await api.download_image(url, output_dir)
                        print(f"Downloaded image {i+1}/{len(download_urls)}")
                    break
                elif status == "error":
                    print("An error occurred during processing.")
                    break
                else:
                    await asyncio.sleep(5)  # Retry after 5 seconds

    except aienlarge.ImgLargerError as e:
        print(f"API Error: {e}")
    except FileNotFoundError:
        print(f"File not found: {image_path}")
    except ValueError as e:
        print(f"Validation Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

> ✅ **Note:** Replace `"path/to/your/image.jpg"` with your actual file path.

---

## 🧰 API Reference

### `ImgLargerAPI` Class

#### `__init__(base_url="https://photoai.imglarger.com/api/PhoAi", username=None)`

- `base_url`: Base URL for the API (default: official endpoint).
- `username`: Optional. Will be loaded from config or generated if not provided.

#### `async upload_image(image_path: str, process_type: int, scale_radio: Optional[int] = None) -> Optional[str]`

Uploads an image to be processed.

- `process_type`:
  - `0`: Default Image Upscaler (supports `scale_radio`)
  - `1`: Sharpen
  - `2`: Enhancer Mode (Photo color and contrast)
  - `3`: Retouch Mode (Deblur)
  - `13`: Anime Images Upscaler (supports `scale_radio`)
- `scale_radio`: Optional scale (only for types `0` and `13`). Valid values: `2`, `4`, `8`.

> ⚠️ **Note:** `scale_radio` with a value of 8 is usually for PRO users only, but using it directly in the API may bypass that.

#### `async check_status(process_code: str) -> Tuple[Optional[str], Optional[List[str]]]`

Returns the current processing status and download URLs (if available).

- `status`: `"pending"`, `"processing"`, `"done"`, `"error"`, or `None`
- `download_urls`: List of URLs when status is `"done"`

#### `async download_image(download_url: str, output_path_dir: str)`

Downloads a processed image to the specified directory.

---

## ❗ Error Handling

Exception classes include:

- `ImgLargerError` (Base)
- `ImgLargerUploadError`
- `ImgLargerStatusError`
- `ImgLargerDownloadError`
- `ImgLargerInvalidProcessTypeError`
- `ImgLargerInvalidScaleRadioError`
- `ImgLargerAPIResponseError`
- `ImgLargerConfigFileError`

These help you implement granular and predictable error handling.

---

## ⚙️ Configuration

- The library stores a username in `.aienlarge_config.json` (created in your working directory).
- If no username is provided, one is generated and saved automatically.

---

## 📚 Dependencies

- [`httpx`](https://www.python-httpx.org/): For async HTTP operations.

---

## 📄 License

Licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

---

## 👤 Author

**SSL-ACTX**
📧 [seuriin@gmail.com](mailto:seuriin@gmail.com)
🔗 [https://github.com/SSL-ACTX](https://github.com/SSL-ACTX)
