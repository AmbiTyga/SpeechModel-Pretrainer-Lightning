import os
import requests

# Function to download files using requests
def download(url):
    try:
        # Extract the filename, removing the token part
        filename = os.path.basename(url).split('?')[0]

        # Send the HTTP GET request
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Save the file
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded: {filename}")
    except requests.RequestException as e:
        print(f"Failed to download: {url}")
        with open(error_file, 'a') as ef:
            ef.write(f"{url}\n")

if __name__ == '__main__':
    url_file = 'indicVoices.train.txt'
    error_file = 'failed_downloads.txt'

    # Clear or create the error file
    with open(error_file, 'w') as ef:
        ef.write('')
    # Read each URL from the file and download it
    with open(url_file, 'r') as uf:
        for line in uf:
            url = line.strip()
            if url:
                download(url)

    print(f"Download completed. Check {error_file} for failed downloads.")
