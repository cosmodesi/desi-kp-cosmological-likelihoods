import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Y1 FS cosmological likelihoods')
    parser.add_argument('--data-dir', type=str, default='./data/likelihood', help='download the data likelihood files from NERSC to this path')
    args = parser.parse_args()

    BASE_URL = "https://data.desi.lbl.gov/public/dr1/vac/dr1/full-shape-bao-clustering/v1.0/data/likelihood/"

    session = requests.Session()

    def is_valid_link(href):
        if not href:
            return False
        if href.startswith("?"):
            return False
        if href in ("../", "/"):
            return False
        return True


    def list_files(url):
        """Recursively list all files under url."""
        response = session.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        files = []

        for link in soup.find_all("a"):
            href = link.get("href")
            if not is_valid_link(href):
                continue

            full_url = urljoin(url, href)

            if href.endswith("/"):
                # Recurse into subdirectory
                files.extend(list_files(full_url))
            else:
                # Skip index pages
                if not href.startswith("index"):
                    files.append(full_url)

        return files


    def download_file(url):
        rel_path = url.replace(BASE_URL, "")
        local_path = os.path.join(args.data_dir, rel_path)

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        with session.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print(f"Downloaded {rel_path}")

    print("Listing files...")
    files = list_files(BASE_URL)
    print(f"Found {len(files)} files")
    for file_url in files:
        download_file(file_url)
    print("Done.")
    print(f"Add data_dir={args.data_dir} under the DESI likelihood name to the cobaya *.yaml config files.")