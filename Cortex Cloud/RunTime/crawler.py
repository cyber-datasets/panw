import requests
import json
import os
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base API endpoints and constants from the Postman collection
BASE_URL = "https://docs-cortex.paloaltonetworks.com"
PRETTY_URL_ENDPOINT = f"{BASE_URL}/internal/api/webapp/pretty-url/reader"
DOCUMENT_MAP_ENDPOINT = f"{BASE_URL}/api/khub/maps/{{document_id}}"
PAGES_ENDPOINT = f"{BASE_URL}/api/khub/maps/{{document_id}}/pages"
CONTENT_ENDPOINT = f"{BASE_URL}/api/khub/maps/{{document_id}}/topics/{{topic_id}}/content"

# Output directory
OUTPUT_DIR = "cortex_docs"
PAGES_DIR = os.path.join(OUTPUT_DIR, "pages")

# Basic HTML template for individual files
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
</head>
<body>
    {content}
</body>
</html>"""

def sanitize_filename(title):
    """Sanitize a title to create a valid filename, preserving spaces and special characters minimally."""
    return re.sub(r'[<>:"/\\|?*]', '_', title).strip()

def count_toc_items(toc):
    """Recursively count all items in the TOC for progress tracking."""
    total = 0
    for item in toc:
        total += 1
        if item["children"]:
            total += count_toc_items(item["children"])
    return total

def fetch_pretty_url(pretty_url):
    """Step 1: Resolve Pretty URL to get documentId and tocId."""
    logger.info(f"Fetching Pretty URL: {pretty_url}")
    payload = {"prettyUrl": pretty_url, "forcedTocId": None}
    response = requests.post(PRETTY_URL_ENDPOINT, json=payload)
    response.raise_for_status()
    data = response.json()
    logger.info(f"Received documentId: {data['documentId']}, tocId: {data['tocId']}")
    return data["documentId"], data["tocId"]

def fetch_document_map(document_id):
    """Step 2: Fetch Document Map to get fingerprint."""
    url = DOCUMENT_MAP_ENDPOINT.format(document_id=document_id)
    logger.info(f"Fetching Document Map for documentId: {document_id}")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    logger.info(f"Received fingerprint: {data['fingerprint']}")
    return data["fingerprint"]

def fetch_pages(document_id, fingerprint):
    """Step 3: Fetch the Table of Contents (TOC)."""
    url = PAGES_ENDPOINT.format(document_id=document_id)
    params = {"v": fingerprint}
    logger.info(f"Fetching TOC for documentId: {document_id} with fingerprint: {fingerprint}")
    response = requests.get(url, params=params)
    response.raise_for_status()
    toc = response.json()["paginatedToc"][0]["pageToc"]
    logger.info(f"TOC fetched with {len(toc)} top-level items")
    return toc

def fetch_content(document_id, topic_id, fingerprint):
    """Step 4: Fetch HTML content for a specific topic."""
    url = CONTENT_ENDPOINT.format(document_id=document_id, topic_id=topic_id)
    params = {"target": "DESIGNED_READER", "v": fingerprint}
    logger.info(f"Fetching content for topicId: {topic_id}")
    response = requests.get(url, params=params)
    response.raise_for_status()
    logger.debug(f"Content fetched for topicId: {topic_id}")
    return response.text

def build_html_structure(toc, document_id, fingerprint, full_html, prefix="", parent_path="", progress_bar=None):
    """Recursively build HTML and file structure with numbered prefixes."""
    for idx, item in enumerate(toc, start=1):
        title = item["title"]
        content_id = item["contentId"]
        topic_level = item.get("topic-level", len(prefix.split('.')) if prefix else 1)
        number_prefix = f"{prefix}{idx}" if prefix else str(idx)
        sanitized_title = sanitize_filename(title)
        numbered_title = f"{number_prefix}_{sanitized_title}"
        current_path = os.path.join(parent_path, numbered_title) if parent_path else numbered_title

        # Fetch content for this section
        html_content = fetch_content(document_id, content_id, fingerprint)
        soup = BeautifulSoup(html_content, 'html.parser')
        content_div = soup.find('div', class_='content-locale-en-US') or soup

        # 1. Append to full HTML with numbered title
        section_html = f"<section id='{content_id}'><h{topic_level}>{number_prefix} {title}</h{topic_level}>{str(content_div)}</section>"
        full_html.append(section_html)

        # 2 & 3. Handle pages and sections with numbered folders/files
        if not parent_path:  # Top-level page
            page_dir = os.path.join(PAGES_DIR, numbered_title)
            os.makedirs(page_dir, exist_ok=True)
            page_file = os.path.join(page_dir, f"{numbered_title}.html")
            logger.info(f"Writing page file: {page_file}")
            with open(page_file, "w", encoding="utf-8") as f:
                f.write(HTML_TEMPLATE.format(title=f"{number_prefix} {title}", content=section_html))
        else:  # Section within a page
            section_dir = os.path.join(PAGES_DIR, parent_path)
            os.makedirs(section_dir, exist_ok=True)
            section_file = os.path.join(section_dir, f"{numbered_title}.html")
            logger.info(f"Writing section file: {section_file}")
            with open(section_file, "w", encoding="utf-8") as f:
                f.write(HTML_TEMPLATE.format(title=f"{number_prefix} {title}", content=html_content))

        # Recurse into children with updated prefix
        if item["children"]:
            build_html_structure(item["children"], document_id, fingerprint, full_html, f"{number_prefix}.", current_path, progress_bar)

        # Update progress bar
        if progress_bar:
            progress_bar.update(1)

def main():
    # Initial setup
    pretty_url = "Cortex-CLOUD/Cortex-Cloud-Runtime-Security-Documentation/Get-started-with-Cortex-Cloud"
    os.makedirs(PAGES_DIR, exist_ok=True)
    logger.info(f"Output directory setup: {OUTPUT_DIR}")

    # Step 1: Get documentId
    document_id, _ = fetch_pretty_url(pretty_url)

    # Step 2: Get fingerprint
    fingerprint = fetch_document_map(document_id)

    # Step 3: Get TOC and count items
    toc = fetch_pages(document_id, fingerprint)
    total_items = count_toc_items(toc)
    logger.info(f"Total TOC items to process: {total_items}")

    # Step 4: Build HTML and file structure with progress bar
    full_html = ["<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><title>Cortex Cloud Runtime Security Documentation</title></head><body>"]
    with tqdm(total=total_items, desc="Processing TOC items") as pbar:
        build_html_structure(toc, document_id, fingerprint, full_html, progress_bar=pbar)
    full_html.append("</body></html>")

    # Write full HTML file
    full_html_file = os.path.join(OUTPUT_DIR, "full_documentation.html")
    logger.info(f"Writing full documentation: {full_html_file}")
    with open(full_html_file, "w", encoding="utf-8") as f:
        f.write("\n".join(full_html))
    logger.info("Documentation generation complete")

if __name__ == "__main__":
    main()