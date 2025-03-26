import os
import requests
import concurrent.futures
from pathlib import Path
import zipfile
import time


def download_file(url, dir):
    """Download a file from URL to specified directory."""
    # Create output directory if it doesn't exist
    os.makedirs(dir, exist_ok=True)

    # Get filename from URL
    filename = os.path.join(dir, url.split("/")[-1])

    # Skip if file already exists
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return filename

    # Download file
    print(f"Downloading {url} to {filename}")
    try:
        start_time = time.time()
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Print progress
                    if total_size > 0:
                        percent = downloaded / total_size * 100
                        print(
                            f"\rProgress: {percent:.1f}% ({downloaded/1_000_000:.1f}MB / {total_size/1_000_000:.1f}MB)",
                            end="",
                        )

        elapsed = time.time() - start_time
        print(f"\nDownload complete! Took {elapsed:.1f} seconds")
        return filename
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return None


def download(urls, dir, threads=4, extract=True):
    """Download and optionally extract multiple URLs in parallel to specified directory."""
    dir_path = Path(dir)
    os.makedirs(dir_path, exist_ok=True)

    # Download files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_url = {
            executor.submit(download_file, url, dir_path): url for url in urls
        }
        downloaded_files = []

        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                filename = future.result()
                if filename:
                    downloaded_files.append(filename)
            except Exception as e:
                print(f"{url} generated an exception: {e}")

    # Extract zip files if requested
    if extract:
        for file in downloaded_files:
            if file.endswith(".zip"):
                print(f"Extracting {file}...")
                try:
                    with zipfile.ZipFile(file, "r") as zip_ref:
                        zip_ref.extractall(dir_path)
                    print(f"Extraction complete: {file}")
                except Exception as e:
                    print(f"Error extracting {file}: {e}")

    return downloaded_files


# Example usage
if __name__ == "__main__":
    dataset_dir = "/home/eyakub/scratch/eecs_project"
    urls = [
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-challenge.zip",
    ]
    download(urls, dataset_dir, threads=4, extract=True)
