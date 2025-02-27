import requests
import json
import os
import re
import shutil
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

def build_section_content(toc, document_id, fingerprint, prefix=""):
    """Build HTML content for a section including all subsections recursively."""
    section_html = []
    for idx, item in enumerate(toc, start=1):
        title = item["title"]
        content_id = item["contentId"]
        number_prefix = f"{prefix}{idx}" if prefix else str(idx)
        html_content = fetch_content(document_id, content_id, fingerprint)
        soup = BeautifulSoup(html_content, 'html.parser')
        content_div = soup.find('div', class_='content-locale-en-US') or soup
        topic_level = item.get("topic-level", len(prefix.split('.')) + 1 if prefix else 1)
        
        # Append current item's content
        section_html.append(f"<section id='{content_id}'><h{topic_level}>{number_prefix} {title}</h{topic_level}>{str(content_div)}</section>")
        
        # Recursively append children content
        if item["children"]:
            section_html.append(build_section_content(item["children"], document_id, fingerprint, f"{number_prefix}."))
    
    return "\n".join(section_html)

def build_html_structure(toc, document_id, fingerprint, full_html, prefix="", parent_path="", progress_bar=None):
    """Build HTML structure with separate subsection files and full section content in parent files."""
    for idx, item in enumerate(toc, start=1):
        title = item["title"]
        content_id = item["contentId"]
        topic_level = item.get("topic-level", len(prefix.split('.')) if prefix else 1)
        number_prefix = f"{prefix}{idx}" if prefix else str(idx)
        sanitized_title = sanitize_filename(title)
        numbered_title = f"{number_prefix}_{sanitized_title}"
        current_path = os.path.join(parent_path, numbered_title) if parent_path else numbered_title

        # Fetch content for this item
        html_content = fetch_content(document_id, content_id, fingerprint)
        soup = BeautifulSoup(html_content, 'html.parser')
        content_div = soup.find('div', class_='content-locale-en-US') or soup

        # Build full content for this section (current item + subsections)
        section_content = build_section_content([item], document_id, fingerprint, prefix)

        # Append to full HTML for the complete documentation
        full_html.append(section_content)

        # Handle file writing based on level
        if not parent_path:  # Top-level section
            page_dir = os.path.join(parent_path, numbered_title) if parent_path else numbered_title
            os.makedirs(page_dir, exist_ok=True)
            page_file = os.path.join(page_dir, f"{numbered_title}.html")
            logger.info(f"Writing top-level section file (with subsections): {page_file}")
            with open(page_file, "w", encoding="utf-8") as f:
                f.write(HTML_TEMPLATE.format(title=f"{number_prefix} {title}", content=section_content))
        else:  # Subsection
            section_dir = parent_path
            os.makedirs(section_dir, exist_ok=True)
            section_file = os.path.join(section_dir, f"{numbered_title}.html")
            logger.info(f"Writing subsection file (with aggregated sub-sections): {section_file}")
            with open(section_file, "w", encoding="utf-8") as f:
                f.write(HTML_TEMPLATE.format(title=f"{number_prefix} {title}", content=section_content))


        # Recurse into children
        if item["children"]:
            build_html_structure(item["children"], document_id, fingerprint, full_html, f"{number_prefix}.", current_path, progress_bar)

        # Update progress bar
        if progress_bar:
            progress_bar.update(1)

def check_existing_files(doc_dir):
    """Check if HTML files exist in the document directory."""
    full_html_file = os.path.join(doc_dir, "full_documentation.html")
    pages_dir = os.path.join(doc_dir, "pages")
    return os.path.exists(full_html_file) or os.path.exists(pages_dir)

def delete_existing_files(doc_dir):
    """Delete existing HTML files and pages directory if they exist."""
    full_html_file = os.path.join(doc_dir, "full_documentation.html")
    pages_dir = os.path.join(doc_dir, "pages")
    if os.path.exists(full_html_file):
        os.remove(full_html_file)
        logger.info(f"Deleted existing file: {full_html_file}")
    if os.path.exists(pages_dir):
        shutil.rmtree(pages_dir)
        logger.info(f"Deleted existing directory: {pages_dir}")

def process_document(pretty_url, product_folder, doc_name, update=False):
    """Process a single document and save it under the product folder."""
    doc_output_dir = os.path.join(OUTPUT_DIR, product_folder, sanitize_filename(doc_name))
    pages_dir = os.path.join(doc_output_dir, "pages")

    # Check update flag and existing files
    if not update and check_existing_files(doc_output_dir):
        logger.info(f"Skipping {doc_name} in {product_folder} as files exist and update is False")
        return
    elif update:
        delete_existing_files(doc_output_dir)

    os.makedirs(pages_dir, exist_ok=True)
    logger.info(f"Processing document: {doc_name} in {doc_output_dir}")

    # Step 1: Get documentId
    document_id, _ = fetch_pretty_url(pretty_url)

    # Step 2: Get fingerprint
    fingerprint = fetch_document_map(document_id)

    # Step 3: Get TOC and count items
    toc = fetch_pages(document_id, fingerprint)
    total_items = count_toc_items(toc)
    logger.info(f"Total TOC items to process for {doc_name}: {total_items}")

    # Step 4: Build HTML and file structure with progress bar
    full_html = [f"<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><title>{doc_name}</title></head><body>"]
    with tqdm(total=total_items, desc=f"Processing {doc_name}") as pbar:
        build_html_structure(toc, document_id, fingerprint, full_html, progress_bar=pbar, parent_path=pages_dir)
    full_html.append("</body></html>")

    # Write full HTML file
    full_html_file = os.path.join(doc_output_dir, "full_documentation.html")
    logger.info(f"Writing full documentation: {full_html_file}")
    with open(full_html_file, "w", encoding="utf-8") as f:
        f.write("\n".join(full_html))
    logger.info(f"Completed processing {doc_name}")

def main():
    # Load the doctree.json file
    with open("doctree.json", "r", encoding="utf-8") as f:
        doctree = json.load(f)

    # Ensure base output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Base output directory setup: {OUTPUT_DIR}")

    # Process each product and its children
    for product in doctree["children"]:
        product_name = sanitize_filename(product["name"])
        logger.info(f"Processing product: {product_name}")
        
        for doc in product["children"]:
            doc_name = doc["name"]
            pretty_url = doc.get("link")
            update = doc.get("update", False)
            if pretty_url:  # Only process if link exists
                process_document(pretty_url, product_name, doc_name, update)

    logger.info("All documentation generation complete")

if __name__ == "__main__":
    main()