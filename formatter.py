import asyncio
import os
from pathlib import Path

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

async def convert_html_recursively(root_dir, out_dir):
    # Ensure the main output directory exists
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build our run_config (non-LLM approach, default Markdown generator)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        word_count_threshold=200,  # Skip extremely small docs if desired
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=0.5,            # Adjust filter sensitivity
                threshold_type="fixed",
                min_word_threshold=50
            )
        ),
        # If your docs have collapsible sections, expand them:
        js_code=["document.querySelectorAll('.ft-expanding-block-link').forEach(link => link.click());"]
    )

    async with AsyncWebCrawler() as crawler:
        # Recursively find all .html files under root_dir
        for html_file in Path(root_dir).rglob("*.html"):
            # Construct file:// URL for the local HTML
            file_url = f"file://{html_file.resolve()}"

            print(f"Processing {html_file.relative_to(root_dir)} ...")

            # Crawl the local HTML file
            result = await crawler.arun(url=file_url, config=run_config)
            if result.success:
                # Build a mirrored output path for the .md
                relative_path = html_file.relative_to(root_dir)
                md_path = out_dir / relative_path.with_suffix(".md")

                # Ensure subdirectories exist in out_dir
                md_path.parent.mkdir(parents=True, exist_ok=True)

                # Write the extracted Markdown
                with md_path.open("w", encoding="utf-8") as f:
                    f.write(result.markdown)

                print(f"  -> Saved Markdown to {md_path}")
            else:
                print(f"  !! Failed: {result.error_message}")

async def main():
    # Base directory under PANW for Cortex products
    base_dir = "cortex_docs"
    
    # List of Cortex products to process
    cortex_products = [
        "Cortex XSIAM/Analytics Alert Reference/pages"
    ]
    
    # Output directory for all Markdown files
    output_root = "PANW/markdown_output"
    
    # Process each Cortex product's pages directory
    for product_path in cortex_products:
        input_root = os.path.join(base_dir, product_path)
        if Path(input_root).exists():
            print(f"Processing {input_root}...")
            await convert_html_recursively(input_root, output_root)
        else:
            print(f"Warning: Directory {input_root} does not exist. Skipping...")

if __name__ == "__main__":
    asyncio.run(main())