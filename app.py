# app.py
import streamlit as st
import os
import re
import json
import io
import pandas as pd
from PIL import Image
import google.generativeai as genai
from io import StringIO
import base64
from urllib.parse import urlparse, urljoin
import requests
import uuid
import zipfile
import time
from pathlib import Path
import traceback # Added for detailed error logging

# --- Import local modules ---
# Ensure these files exist and are importable
try:
    from getImages import screenshot_tables
    from amazonTables import amazon_tables
except ImportError as e:
    st.error(f"Error importing local modules: {e}. Make sure 'getImages.py' and 'amazonTables.py' are in the same directory.")
    st.stop()

# --- API Key Configuration ---
GEMINI_API_KEY = os.getenv("API_KEY") # Replace with your actual API key

# Use Streamlit secrets if available, otherwise fall back to environment variable
if not GEMINI_API_KEY and 'GEMINI_API_KEY' in st.secrets:
    GEMINI_API_KEY = st.secrets['GEMINI_API_KEY']

if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
    st.warning("Gemini API Key not found or not set. Please set it using environment variables or Streamlit secrets (key: GEMINI_API_KEY). AI features will be disabled.", icon="⚠️")
    ai_enabled = False
    model = None # No model if AI is disabled
else:
    ai_enabled = True
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Initialize the model only once if AI is enabled
        # Using gemini-1.5-flash as it's generally faster and capable for this
        model = genai.GenerativeModel('gemini-1.5-flash')
        st.success("Gemini API Key configured successfully. AI features enabled.", icon="✅")
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}. AI features disabled.", icon="❌")
        ai_enabled = False
        model = None

# --- Page Setup ---
st.set_page_config(page_title="Web Table Extractor", layout="wide")
st.title("🌐 Web Table Extractor")
st.markdown("Extract data from tables on webpages. Handles Amazon product comparisons directly, uses HTML parsing or AI (Gemini Vision) for others.")

# Initialize session state variables
default_session_state = {
    'extraction_method': None,
    'data_extracted': False,
    'extracted_tables': [],
    'extracted_formats': [],
    'extracted_filenames': [],
    'screenshot_captured': False, # Indicates if any screenshots were taken
    'screenshot_filenames': [], # Stores paths to all screenshots captured in a run
    'selected_table_index': 0,
    'downloaded_images': [], # Stores paths to images downloaded from table cells
    'all_downloaded_image_paths': set() # Keep track of all unique downloaded image paths across tables
}
for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Helper Functions ---

def get_download_link(content, filename, text, format_type):
    """Creates a download link for text content."""
    if isinstance(content, bytes):
        b64 = base64.b64encode(content).decode()
    elif isinstance(content, str):
         b64 = base64.b64encode(content.encode('utf-8')).decode() # Ensure UTF-8 for strings
    else:
        # Attempt to convert other types to string
        try:
            b64 = base64.b64encode(str(content).encode('utf-8')).decode()
        except Exception as e:
            st.error(f"Error encoding content for download link ({type(content)}): {e}")
            return f"<span>Error creating link for {filename}</span>"

    mime_type = {
        "CSV": "text/csv",
        "JSON": "application/json",
        "HTML": "text/html",
        "TEXT": "text/plain" # Added for potential Gemini fallback/error text
    }.get(format_type, "application/octet-stream")

    # Ensure filename is filesystem-safe (basic sanitation)
    safe_filename = re.sub(r'[\\/*?:"<>|]', "_", filename)

    href = f'<a href="data:{mime_type};charset=utf-8;base64,{b64}" download="{safe_filename}">{text}</a>'
    return href

def extract_image_url(cell_val):
    """Extract image URL from HTML cell value using multiple patterns."""
    if not isinstance(cell_val, str): # Ensure input is a string
        return None

    # Prioritize data-a-hires, data-src often found in modern sites/Amazon
    patterns = [
        r'<img[^>]*?data-a-hires=[\'"]([^\'"]+)[\'"]',
        r'<img[^>]*?data-src=[\'"]([^\'"]+)[\'"]',
        r'<img[^>]*?src=[\'"]([^\'"]+)[\'"]',
        # Last resort: more generic src-like attribute
        r'<img[^>]*?(?:src|data-[^=]+)=[\'"]([^\'"]+)[\'"]'
    ]
    for pattern in patterns:
        match = re.search(pattern, cell_val)
        if match:
            return match.group(1)
    return None

def resolve_relative_url(base_url, img_url):
    """Resolve a relative URL to absolute URL."""
    if not img_url:
        return None
    try:
        # If already absolute or a data URI, return as is
        if urlparse(img_url).scheme or img_url.startswith('data:'):
            return img_url
        # Use urljoin for robust relative path resolution
        return urljoin(base_url, img_url)
    except Exception as e:
        print(f"Error resolving URL '{img_url}' with base '{base_url}': {e}")
        return img_url # Return original as fallback

def download_image(img_url, img_filename, timeout=15, max_retries=2):
    """Download image with retry logic and better error handling."""
    if not img_url:
        return False

    # Ensure directory exists
    try:
        os.makedirs(os.path.dirname(img_filename), exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {os.path.dirname(img_filename)}: {e}")
        return False

    # Handle data URIs
    if img_url.startswith('data:image/'):
        try:
            header, encoded = img_url.split(',', 1)
            if ';base64' in header:
                img_data = base64.b64decode(encoded)
                with open(img_filename, 'wb') as img_file:
                    img_file.write(img_data)
                return True
            else:
                print(f"Unsupported data URI format: {header}")
                return False
        except Exception as data_uri_err:
            print(f"Error processing data URI: {data_uri_err}")
            return False

    # Handle regular URL download
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                # Attempt to derive referer, handle potential parsing errors
                'Referer': urlparse(img_url).scheme + '://' + urlparse(img_url).netloc if urlparse(img_url).netloc else ''
            }
            response = requests.get(img_url, headers=headers, stream=True, timeout=timeout)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            content_type = response.headers.get('Content-Type', '').lower()
            if 'image' not in content_type:
                 print(f"Warning: URL {img_url} did not return an image (Content-Type: {content_type}). Skipping.")
                 return False # Don't retry if content type is wrong

            with open(img_filename, 'wb') as img_file:
                for chunk in response.iter_content(chunk_size=8192):
                    img_file.write(chunk)

            if os.path.exists(img_filename) and os.path.getsize(img_filename) > 0:
                return True
            else:
                print(f"Downloaded image file is empty or non-existent: {img_filename}")
                # Optionally remove the empty file: os.remove(img_filename)
                return False # Treat empty download as failure

        except requests.exceptions.RequestException as e:
            print(f"Error downloading image {img_url} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1) # Wait before retrying
            else:
                print(f"Failed to download image after {max_retries} attempts: {img_url}")
                return False
        except Exception as e: # Catch other potential errors
            print(f"Unexpected error downloading {img_url}: {e}")
            return False

    return False # Should not be reached if retries complete

def process_table_images(table_data, base_url, image_dir="extracted_images"):
    """
    Process images found in HTML DataFrame cells.
    Downloads them and updates the DataFrame *copy* with local paths.
    This should ONLY be run on DataFrames derived directly from HTML parsing.
    """
    # Ensure we have a DataFrame and it potentially has images based on HTML parsing
    if 'dataframe' not in table_data or table_data['dataframe'] is None:
        return table_data
    if not any("<img" in str(cell) for row in table_data['dataframe'].values for cell in row):
        table_data['has_images'] = False
        return table_data

    # Mark that the original HTML likely contained images
    table_data['has_images'] = True

    # Create image directory if it doesn't exist
    os.makedirs(image_dir, exist_ok=True)

    # Generate a safe base filename using the table title and domain
    url_domain = urlparse(base_url).netloc.replace("www.", "")
    safe_domain = re.sub(r'[^\w-]', '', url_domain).strip().replace('.', '_')
    safe_title = re.sub(r'[^\w\s-]', '', table_data.get('title', 'table')).strip().replace(' ', '_')
    safe_title = re.sub(r'[-]+', '_', safe_title)
    if not safe_title: safe_title = "table" # Fallback title

    # Clone the DataFrame to modify
    df_copy = table_data['dataframe'].copy()
    processed = False
    table_downloaded_images = []

    for i, row in df_copy.iterrows():
        for col in df_copy.columns:
            cell_val = str(row[col]) # Work with string representation
            if "<img" in cell_val:
                img_url = extract_image_url(cell_val)
                if img_url:
                    absolute_img_url = resolve_relative_url(base_url, img_url)
                    if absolute_img_url:
                        # Create a unique filename
                        try:
                            file_ext = os.path.splitext(urlparse(absolute_img_url).path)[-1].lower()
                            if not file_ext or len(file_ext) > 5 or file_ext == '.': # Basic validation
                                # Guess extension from data URI or default
                                if absolute_img_url.startswith('data:image/'):
                                    mime_type = absolute_img_url.split(';')[0].split('/')[-1]
                                    file_ext = f".{mime_type}" if mime_type else ".jpg"
                                else:
                                    file_ext = '.jpg' # Default extension
                        except Exception:
                             file_ext = '.jpg' # Fallback on parsing error

                        # Limit title length in filename
                        short_safe_title = safe_title[:20]
                        img_filename_base = f"{safe_domain}_{short_safe_title}_{uuid.uuid4().hex[:6]}"
                        img_filename = os.path.join(image_dir, f"{img_filename_base}{file_ext}")

                        # Download the image
                        if download_image(absolute_img_url, img_filename):
                            # Update cell with local image path (relative for potentially easier zipping)
                            relative_img_path = os.path.join(os.path.basename(image_dir), os.path.basename(img_filename))
                            df_copy.at[i, col] = f"Image: {relative_img_path}"
                            table_downloaded_images.append(img_filename) # Store full path for zipping
                            processed = True
                        else:
                            df_copy.at[i, col] = f"Image URL: {absolute_img_url} (download failed)"
                            processed = True
                    else:
                        df_copy.at[i, col] = f"Image tag found but URL could not be resolved: {img_url}"
                        processed = True
                else:
                    # Keep original if URL extraction failed but tag exists
                    # df_copy.at[i, col] = f"Image tag found but URL could not be extracted"
                    pass # Or keep original cell content

    # Update table data if images were processed
    if processed:
        table_data['dataframe_with_local_images'] = df_copy
        table_data['downloaded_images'] = table_downloaded_images # Store paths specific to this table
        st.session_state['all_downloaded_image_paths'].update(table_downloaded_images) # Add to global set

    return table_data


# --- Input Section ---
url = st.text_input("Enter the URL of the webpage:", key="url_input", placeholder="e.g., https://www.example.com/page_with_tables")

# Determine page type and set UI accordingly
is_amazon_product_page = bool(url and ("amazon." in urlparse(url).netloc or "amzn." in urlparse(url).netloc) and ("/dp/" in url or "/gp/product/" in url))

use_ai_extraction = False
if not is_amazon_product_page and ai_enabled:
    use_ai_extraction = st.checkbox("✨ Use AI (Gemini) to extract tables from **screenshots** (Beta)", value=False,
                                    help="Requires taking screenshots. May be slower but can extract tables missed by HTML parsing or within images. Disables table image download.")
elif not is_amazon_product_page and not ai_enabled:
    st.info("AI features disabled (API key missing). Using standard HTML table parsing.", icon="ℹ️")


if is_amazon_product_page:
    st.info("Amazon product page detected. Direct extraction will be attempted.", icon="🛒")
    button_text = "Extract Amazon Tables"
    extraction_type_info = "Amazon Direct"
else:
    if use_ai_extraction:
        st.info("General webpage. AI (Gemini Vision) extraction selected. Will analyze screenshots.", icon="🤖")
        button_text = "Extract Tables using AI"
        extraction_type_info = "AI (Screenshot)"
    else:
        st.info("General webpage. Standard HTML table extraction selected.", icon="📄")
        button_text = "Extract Tables using HTML"
        extraction_type_info = "HTML Parsing"


format_type = st.selectbox(
    "Select desired output format:",
    options=["CSV", "JSON", "HTML"],
    index=0 # Default to CSV
)

# --- Action Button ---
if st.button(button_text, key="main_action_button", disabled=not url, use_container_width=True):
    # Reset state before action
    st.session_state.update({
        'data_extracted': False,
        'extracted_tables': [],
        'extracted_formats': [],
        'extracted_filenames': [],
        'screenshot_captured': False,
        'screenshot_filenames': [],
        'extraction_method': None,
        'selected_table_index': 0,
        'downloaded_images': [], # Deprecated? Let's clear it anyway.
        'all_downloaded_image_paths': set() # Reset collected image paths
    })

    progress = st.progress(0)
    status = st.empty()

    if is_amazon_product_page:
        # --- Amazon Direct Extraction Workflow ---
        st.session_state.extraction_method = 'amazon'
        status.text("Attempting direct extraction from Amazon...")
        progress.progress(25)

        try:
            # Call the imported function
            extracted_result = amazon_tables(url)
            progress.progress(75)

            # Check the status returned by the function
            if extracted_result.get('status') == 'success':
                status.success("Amazon table extracted successfully!")
                df = extracted_result.get("dataframe")
                html_content = extracted_result.get("html", "") # Use original HTML for display/download

                if df is None:
                     st.error("Amazon extraction reported success but no DataFrame was returned.")
                     st.stop()

                # Save as a table entry
                table_data = {
                    "title": "Amazon Product Comparison",
                    "dataframe": df, # This is the parsed data
                    "html_source": html_content, # Keep original HTML source
                    "source_url": url,
                    "extraction_type": "Amazon Direct",
                    "screenshot": None, # No screenshot for direct Amazon extraction
                    "has_images": False, # Will be updated by process_table_images if needed
                    "ai_extraction_failed": False,
                }

                # Process images IF they exist in the HTML-derived dataframe
                status.text("Processing images in the Amazon table (if any)...")
                table_data = process_table_images(table_data, url)
                st.session_state['all_downloaded_image_paths'].update(table_data.get('downloaded_images', []))


                # Format table data based on the CORE dataframe (df)
                # The 'dataframe_with_local_images' is for a secondary view/download
                filename_base = "amazon_comparison"
                if format_type == "CSV":
                    table_data["content"] = df.to_csv(index=False)
                    table_data["format"] = "CSV"
                    table_data["filename"] = f"{filename_base}.csv"
                    if "dataframe_with_local_images" in table_data:
                       table_data["content_with_local_images"] = table_data["dataframe_with_local_images"].to_csv(index=False)
                       table_data["filename_local_images"] = f"{filename_base}_local_images.csv"
                elif format_type == "JSON":
                    table_data["content"] = df.to_json(orient="records", indent=2)
                    table_data["format"] = "JSON"
                    table_data["filename"] = f"{filename_base}.json"
                    if "dataframe_with_local_images" in table_data:
                        table_data["content_with_local_images"] = table_data["dataframe_with_local_images"].to_json(orient="records", indent=2)
                        table_data["filename_local_images"] = f"{filename_base}_local_images.json"
                else:  # HTML - use original HTML source if available
                    table_data["content"] = f"<html><head><meta charset='UTF-8'><title>Amazon Comparison</title></head><body>\n{html_content}\n</body></html>" if html_content else df.to_html(escape=False, index=False)
                    table_data["format"] = "HTML"
                    table_data["filename"] = f"{filename_base}.html"
                    if "dataframe_with_local_images" in table_data:
                        local_html = table_data["dataframe_with_local_images"].to_html(escape=False, index=False)
                        table_data["content_with_local_images"] = f"<html><head><meta charset='UTF-8'><title>Amazon Comparison (Local Images)</title></head><body>\n{local_html}\n</body></html>"
                        table_data["filename_local_images"] = f"{filename_base}_local_images.html"


                st.session_state.extracted_tables.append(table_data)
                st.session_state.extracted_formats.append(format_type) # Keep track? maybe redundant
                st.session_state.extracted_filenames.append(table_data["filename"]) # Keep track? maybe redundant
                st.session_state.data_extracted = True
                progress.progress(100)

            else:
                # Handle different failure statuses from amazon_tables
                error_msg = extracted_result.get('message', 'Unknown error during Amazon extraction.')
                status.error(f"Amazon Extraction Failed: {error_msg}")
                progress.progress(100)
                st.session_state.data_extracted = False

        except Exception as e:
            status.error(f"An unexpected error occurred during Amazon extraction: {e}")
            st.expander("Traceback").code(traceback.format_exc())
            progress.progress(100)
            st.session_state.data_extracted = False

    else:
        # --- General Webpage Table Extraction Workflow (HTML or AI) ---
        st.session_state.extraction_method = 'general_ai' if use_ai_extraction else 'general_html'
        status.text("Preparing to extract tables...")
        progress.progress(5)

        try:
            # --- Step 1: Get Table Info (Screenshots and HTML) ---
            status.text("Taking screenshots and parsing HTML for tables...")
            progress.progress(15)
            tables_info = screenshot_tables(url) # This function needs to return screenshot paths and potentially parsed dataframes/html
            progress.progress(30)

            if not tables_info:
                status.warning("No tables found on the webpage or failed to capture.")
                progress.progress(100)
                st.warning("No tables were detected or captured from the provided webpage. Try a different URL or check browser automation setup.")
                st.stop() # Stop execution if no tables found

            status.text(f"Found {len(tables_info)} potential table areas. Processing...")
            # Collect all screenshot paths for potential download later
            st.session_state.screenshot_filenames = [info.get("screenshot") for info in tables_info if info.get("screenshot") and os.path.exists(info.get("screenshot"))]
            st.session_state.screenshot_captured = bool(st.session_state.screenshot_filenames)


            # Prepare common elements for filenames
            extraction_id = uuid.uuid4().hex[:8]
            domain_name = urlparse(url).netloc.replace("www.", "")
            safe_domain = re.sub(r'[^\w-]', '', domain_name).strip().replace('.', '_')


            # --- Step 2: Process Each Table (AI or HTML) ---
            successful_extractions = 0
            for idx, table_info in enumerate(tables_info):
                current_table_progress = 30 + int(65 * (idx + 1) / len(tables_info))
                progress.progress(current_table_progress)
                status.text(f"Processing table {idx+1} of {len(tables_info)}...")

                df = None # Initialize DataFrame for this table
                extraction_type_for_table = "HTML Parsing" # Default
                ai_extraction_failed = False
                html_source_for_table = table_info.get("html", "") # Get original HTML if available
                screenshot_path = table_info.get("screenshot")
                table_title = table_info.get("title", f"Table {idx+1}")
                safe_title = re.sub(r'[^\w\s-]', '', table_title).strip().replace(' ', '_')
                safe_title = re.sub(r'[-]+', '_', safe_title)
                if not safe_title: safe_title = f"table_{idx+1}"
                filename_base = f"{safe_domain}_{safe_title[:30]}_{extraction_id}" # Shortened title


                # --- Attempt AI Extraction if selected and possible ---
                if use_ai_extraction and ai_enabled and model and screenshot_path and os.path.exists(screenshot_path):
                    extraction_type_for_table = "AI (Screenshot)"
                    status.text(f"Processing table {idx+1} with AI (Gemini)...")
                    try:
                        img = Image.open(screenshot_path)
                        # More robust prompt
                        prompt = f"""Analyze the following image, which contains a data table.
                        Extract the complete data from the table.
                        Format the output STRICTLY as a CSV (Comma Separated Values) string.
                        - Use commas (,) as delimiters.
                        - Enclose fields containing commas, quotes, or newlines in double quotes (").
                        - Escape double quotes within fields using two double quotes ("").
                        - Include a header row based on the table structure in the image. If no header is visible, generate generic headers like 'Column 1', 'Column 2', etc.
                        - Ensure each row from the table corresponds to a new line in the CSV.
                        - Output ONLY the raw CSV data. Do not include any introductory text, explanations, summaries, or markdown code fences like ```csv ... ```."""

                        # Generate content with timeout
                        response = model.generate_content([prompt, img], request_options={"timeout": 120}) # Increased timeout

                        # Attempt to parse the response as CSV
                        csv_data = response.text.strip()

                        # Basic check if the response looks like CSV (e.g., contains commas or newlines)
                        if ',' not in csv_data and '\n' not in csv_data and len(csv_data.split()) > 5: # Avoid treating short error messages as CSV
                             raise ValueError("AI response did not resemble CSV data.")
                        if "```" in csv_data: # Try to remove markdown fences if present
                             csv_data = re.sub(r'^```[a-z]*\n', '', csv_data, flags=re.MULTILINE)
                             csv_data = re.sub(r'\n```$', '', csv_data, flags=re.MULTILINE)

                        # Use StringIO to read the CSV string into pandas
                        string_io = StringIO(csv_data)
                        df = pd.read_csv(string_io, sep=',', quotechar='"', skipinitialspace=True) # Use pandas robust CSV parsing

                        # Check if DataFrame is empty or has only headers
                        if df.empty or (df.shape[0] == 1 and df.iloc[0].isnull().all()):
                           raise ValueError("AI extracted an empty or header-only table.")

                        status.text(f"Table {idx+1} AI extraction successful.")
                        html_source_for_table = f"<!-- AI Extracted Data from Screenshot: {os.path.basename(screenshot_path)} -->\n" + df.to_html(index=False, escape=True) # Generate basic HTML from AI data

                    except Exception as ai_err:
                        status.warning(f"AI extraction failed for table {idx+1}: {ai_err}. Attempting fallback to HTML parsing.")
                        print(f"AI Error details for table {idx+1}: {traceback.format_exc()}")
                        ai_extraction_failed = True
                        df = None # Ensure df is None if AI fails

                # --- Fallback or Default: Use HTML Parsing ---
                if df is None: # If AI was not used, or AI failed
                    if ai_extraction_failed:
                        extraction_type_for_table = "HTML Parsing (AI Fallback)"
                    else:
                         extraction_type_for_table = "HTML Parsing"

                    status.text(f"Processing table {idx+1} using HTML parsing...")
                    df = table_info.get("dataframe") # Get the pre-parsed DataFrame from getImages
                    if df is None:
                        status.warning(f"No DataFrame found from HTML parsing for table {idx+1}.")
                        # Optionally try parsing html_source_for_table here if needed
                        # try:
                        #     dfs = pd.read_html(StringIO(html_source_for_table))
                        #     if dfs: df = dfs[0]
                        # except: pass
                        if df is None:
                            status.warning(f"Could not extract table {idx+1} via HTML either. Skipping.")
                            continue # Skip this table if no data obtained
                    else:
                         status.text(f"Table {idx+1} HTML parsing successful.")


                # --- Prepare Table Data Object ---
                table_data = {
                    "title": table_title,
                    "dataframe": df, # The primary DataFrame (from AI or HTML)
                    "html_source": html_source_for_table, # Original HTML or AI-generated HTML
                    "source_url": url,
                    "extraction_type": extraction_type_for_table,
                    "screenshot": screenshot_path,
                    "has_images": False, # Assume false unless process_table_images updates it
                    "ai_extraction_failed": ai_extraction_failed,
                }

                # --- Process Images only if using HTML parsing AND images might exist ---
                # Skip image processing if AI was used (df won't have img tags) or if HTML df is None
                if extraction_type_for_table.startswith("HTML") and df is not None:
                    status.text(f"Processing images in table {idx+1} (if any)...")
                    table_data = process_table_images(table_data, url) # Updates 'has_images', adds 'dataframe_with_local_images', 'downloaded_images'
                    st.session_state['all_downloaded_image_paths'].update(table_data.get('downloaded_images', []))


                # --- Format Content based on the main DataFrame (df) ---
                if format_type == "CSV":
                    table_data["content"] = df.to_csv(index=False)
                    table_data["format"] = "CSV"
                    table_data["filename"] = f"{filename_base}.csv"
                    if "dataframe_with_local_images" in table_data:
                        table_data["content_with_local_images"] = table_data["dataframe_with_local_images"].to_csv(index=False)
                        table_data["filename_local_images"] = f"{filename_base}_local_images.csv"
                elif format_type == "JSON":
                    table_data["content"] = df.to_json(orient="records", indent=2)
                    table_data["format"] = "JSON"
                    table_data["filename"] = f"{filename_base}.json"
                    if "dataframe_with_local_images" in table_data:
                        table_data["content_with_local_images"] = table_data["dataframe_with_local_images"].to_json(orient="records", indent=2)
                        table_data["filename_local_images"] = f"{filename_base}_local_images.json"
                else:  # HTML
                    # Prefer original HTML source if available and HTML parsing was used
                    if extraction_type_for_table.startswith("HTML") and table_data["html_source"] and not table_data["html_source"].startswith("<!-- AI Extracted"):
                         display_html = table_data["html_source"]
                    else: # Otherwise, generate from df
                         display_html = df.to_html(escape=False, index=False) # Use escape=False cautiously
                    table_data["content"] = f"<html><head><meta charset='UTF-8'><title>{table_title}</title></head><body>\n{display_html}\n</body></html>"
                    table_data["format"] = "HTML"
                    table_data["filename"] = f"{filename_base}.html"
                    if "dataframe_with_local_images" in table_data:
                        local_html = table_data["dataframe_with_local_images"].to_html(escape=False, index=False)
                        table_data["content_with_local_images"] = f"<html><head><meta charset='UTF-8'><title>{table_title} (Local Images)</title></head><body>\n{local_html}\n</body></html>"
                        table_data["filename_local_images"] = f"{filename_base}_local_images.html"

                # Add successfully processed table to session state
                st.session_state.extracted_tables.append(table_data)
                successful_extractions += 1

            # --- Finalize General Extraction ---
            progress.progress(100)
            if successful_extractions > 0:
                st.session_state.data_extracted = True
                status.success(f"Successfully extracted {successful_extractions} out of {len(tables_info)} potential tables!")
            else:
                status.error("Extraction process completed, but no tables could be successfully extracted.")
                st.session_state.data_extracted = False

        except Exception as e:
            status.error(f"An error occurred during general table extraction: {str(e)}")
            st.expander("Traceback").code(traceback.format_exc())
            progress.progress(100)
            st.session_state.data_extracted = False


    # Rerun only if data was extracted or a significant error occurred that needs display
    if st.session_state.data_extracted or not progress._progress == 100: # If progress didn't complete, likely an error to show
         st.rerun()
    elif not st.session_state.data_extracted: # If no data, ensure final status message is shown
        pass # Don't rerun, let the error messages stay.


# --- Display Results and Download Links ---
if st.session_state.data_extracted and st.session_state.extracted_tables:
    st.markdown("---")
    st.header(f"📊 Extracted Tables ({len(st.session_state.extracted_tables)})")

    # Table selection if multiple tables
    if len(st.session_state.extracted_tables) > 1:
        table_titles = [f"{idx+1}. {table.get('title', f'Table {idx+1}')} ({table.get('extraction_type', 'N/A')})" for idx, table in enumerate(st.session_state.extracted_tables)]
        selected_table_index = st.selectbox(
            "Select a table to view:",
            options=range(len(table_titles)),
            format_func=lambda x: table_titles[x],
            index=st.session_state.selected_table_index,
            key="table_selector"
        )
        # Update session state immediately if selection changes
        if selected_table_index != st.session_state.selected_table_index:
             st.session_state.selected_table_index = selected_table_index
             st.rerun() # Rerun to update display for the new table
    else:
        selected_table_index = 0
        st.session_state.selected_table_index = 0 # Ensure it's set

    # Get the selected table data
    if selected_table_index < len(st.session_state.extracted_tables):
        table_data = st.session_state.extracted_tables[selected_table_index]
    else:
        st.warning("Selected table index is out of bounds. Resetting to the first table.")
        st.session_state.selected_table_index = 0
        table_data = st.session_state.extracted_tables[0]


    # Display table information
    st.subheader(f"{table_data.get('title', 'Extracted Table')}")
    st.caption(f"Extraction Method: {table_data.get('extraction_type', 'Unknown')} | Source URL: {table_data.get('source_url', 'N/A')}")
    if table_data.get('ai_extraction_failed', False):
        st.warning("AI extraction was attempted but failed for this table; showing HTML fallback.", icon="⚠️")

    # Display screenshot if available
    screenshot_path = table_data.get("screenshot")
    if screenshot_path and os.path.exists(screenshot_path):
        with st.expander("View Table Screenshot", expanded=False):
            try:
                image = Image.open(screenshot_path)
                st.image(image, caption=f"Screenshot used for AI extraction or context: {table_data.get('title', '')}", use_container_width=True)
            except Exception as img_e:
                st.error(f"Error loading screenshot image: {img_e}")
    elif table_data.get('extraction_type', '').startswith("AI"):
         st.warning("AI extraction was used, but the associated screenshot path is missing or invalid.", icon="⚠️")


    # Display the main dataframe
    df = table_data.get("dataframe")
    if df is not None and not df.empty:
        st.dataframe(df)
        st.caption(f"Table Preview ({df.shape[0]} rows, {df.shape[1]} columns)")

        # Display raw content
        with st.expander(f"View Raw Extracted {table_data.get('format', 'Data')}"):
            raw_content = table_data.get('content', 'No raw content available.')
            fmt = table_data.get('format', 'TEXT').lower()
            try:
                if fmt == "html":
                    st.code(raw_content, language='html')
                elif fmt == "json":
                     # Try parsing JSON for better display, fallback to text
                     try:
                         st.json(json.loads(raw_content))
                     except json.JSONDecodeError:
                         st.text(raw_content)
                elif fmt == "csv":
                    st.text(raw_content)
                else:
                    st.text(raw_content)
            except Exception as display_err:
                 st.error(f"Error displaying raw content: {display_err}")
                 st.text(raw_content) # Fallback to plain text


        # Display DataFrame with local image paths if available (only from HTML parsing)
        if "dataframe_with_local_images" in table_data:
            with st.expander("View Table With Local Image Paths (from HTML source)"):
                st.dataframe(table_data["dataframe_with_local_images"])
                st.caption("Table with local file paths to downloaded images (derived from original HTML).")

                 # Show downloaded images preview specific to this table
                table_images = table_data.get("downloaded_images", [])
                if table_images:
                    st.write("**Downloaded Images for this table:**")
                    cols = st.columns(min(5, len(table_images))) # Show more previews if space allows
                    for i, img_path in enumerate(table_images):
                        if i >= 10: # Limit preview count
                             st.caption(f"...and {len(table_images) - 10} more images.")
                             break
                        try:
                            if os.path.exists(img_path):
                                with cols[i % 5]:
                                    img = Image.open(img_path)
                                    st.image(img, caption=os.path.basename(img_path), width=100) # Smaller previews
                            else:
                                 with cols[i % 5]:
                                     st.warning(f"Missing: {os.path.basename(img_path)}")
                        except Exception as img_prev_err:
                             print(f"Error displaying image preview {img_path}: {img_prev_err}")
                             with cols[i % 5]:
                                 st.error(f"Error: {os.path.basename(img_path)}")
                else:
                     st.info("No images were successfully downloaded for this specific table.")

    elif df is not None and df.empty:
         st.warning("The extracted table appears to be empty.")
    else:
        st.error("No DataFrame could be displayed for this table.")


    # --- Download Section ---
    st.markdown("---")
    st.subheader("⬇️ Download Options")

    col1, col2, col3 = st.columns(3)

    # Download main extracted data
    with col1:
        if 'content' in table_data and 'filename' in table_data and 'format' in table_data:
            try:
                download_link = get_download_link(
                    table_data['content'],
                    table_data['filename'],
                    f"Download as {table_data['format']} ({table_data['filename']})",
                    table_data['format']
                )
                st.markdown(download_link, unsafe_allow_html=True)
            except Exception as dl_error:
                st.error(f"Error generating download link: {dl_error}")
        else:
            st.caption("Main download not available.")

    # Download version with local image paths (if available)
    with col2:
        if "content_with_local_images" in table_data and "filename_local_images" in table_data:
            try:
                download_link_local = get_download_link(
                    table_data['content_with_local_images'],
                    table_data['filename_local_images'],
                    f"Download {table_data['format']} with local paths ({table_data['filename_local_images']})",
                    table_data['format']
                )
                st.markdown(download_link_local, unsafe_allow_html=True)
            except Exception as dl_error_local:
                st.error(f"Error generating local images download link: {dl_error_local}")
        else:
            st.caption("Local image path version not applicable or available.")

    # Download images for the *current* table (if any were downloaded)
    with col3:
        current_table_images = table_data.get("downloaded_images", [])
        if current_table_images:
            try:
                zip_buffer_current = io.BytesIO()
                with zipfile.ZipFile(zip_buffer_current, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    added_count = 0
                    for img_path in current_table_images:
                        if os.path.exists(img_path):
                            zipf.write(img_path, os.path.join("images", os.path.basename(img_path))) # Put in 'images' subfolder
                            added_count +=1
                        else:
                             print(f"Skipping missing image for current table zip: {img_path}")

                if added_count > 0:
                    zip_buffer_current.seek(0)
                    safe_title_zip = re.sub(r'[^\w-]', '', table_data.get('title', 'table')).strip().replace(' ', '_')
                    zip_filename_current = f"{safe_title_zip}_images.zip"
                    st.download_button(
                        label=f"Download {added_count} Images (ZIP)",
                        data=zip_buffer_current,
                        file_name=zip_filename_current,
                        mime="application/zip",
                        key=f"download_images_{selected_table_index}" # Unique key per table
                    )
                else:
                     st.caption("No downloadable images found.")

            except Exception as zip_error:
                st.error(f"Error creating image ZIP: {zip_error}")
        else:
            st.caption("No images downloaded for this table.")


    # --- Download All Option ---
    st.markdown("---")
    st.subheader("📦 Download All Extracted Data")

    # Consolidate items to zip
    all_files_to_zip = []
    all_images_to_zip = set(st.session_state.get('all_downloaded_image_paths', set()))
    all_screenshots_to_zip = set(st.session_state.get('screenshot_filenames', []))

    for idx, table in enumerate(st.session_state.extracted_tables):
        # Add main content file
        if 'content' in table and 'filename' in table:
            all_files_to_zip.append({'filename': table['filename'], 'content': table['content']})
        # Add content with local images file if exists
        if 'content_with_local_images' in table and 'filename_local_images' in table:
             all_files_to_zip.append({'filename': table['filename_local_images'], 'content': table['content_with_local_images']})

    if all_files_to_zip or all_images_to_zip or all_screenshots_to_zip:
        if st.button("Prepare ZIP file with all tables, images, and screenshots", use_container_width=True, key="download_all_zip"):
            try:
                zip_buffer_all = io.BytesIO()
                with zipfile.ZipFile(zip_buffer_all, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add table data files
                    for file_info in all_files_to_zip:
                        try:
                           # Ensure content is bytes for writestr
                           content_bytes = file_info['content'].encode('utf-8') if isinstance(file_info['content'], str) else bytes(file_info['content'])
                           zipf.writestr(file_info['filename'], content_bytes)
                        except Exception as write_err:
                           print(f"Error writing file {file_info['filename']} to zip: {write_err}")
                           zipf.writestr(f"ERROR_{file_info['filename']}.txt", f"Error writing file: {write_err}".encode('utf-8'))


                    # Add downloaded images (from table cells) into an 'images' folder
                    added_image_count = 0
                    if all_images_to_zip:
                         img_folder = "downloaded_images"
                         for img_path in all_images_to_zip:
                             if os.path.exists(img_path):
                                 zipf.write(img_path, os.path.join(img_folder, os.path.basename(img_path)))
                                 added_image_count += 1
                             else:
                                 print(f"Skipping missing image for all zip: {img_path}")

                    # Add screenshots into a 'screenshots' folder
                    added_screenshot_count = 0
                    if all_screenshots_to_zip:
                        ss_folder = "screenshots"
                        for ss_path in all_screenshots_to_zip:
                             if os.path.exists(ss_path):
                                 zipf.write(ss_path, os.path.join(ss_folder, os.path.basename(ss_path)))
                                 added_screenshot_count += 1
                             else:
                                  print(f"Skipping missing screenshot for all zip: {ss_path}")

                # Prepare download button
                zip_buffer_all.seek(0)
                url_domain = urlparse(url).netloc.replace("www.", "")
                safe_domain_zip = re.sub(r'[^\w-]', '', url_domain).strip().replace('.', '_')
                zip_filename_all = f"{safe_domain_zip}_all_extracted_data.zip"

                st.download_button(
                    label=f"Download All ({len(all_files_to_zip)} files, {added_image_count} images, {added_screenshot_count} screenshots) as ZIP",
                    data=zip_buffer_all,
                    file_name=zip_filename_all,
                    mime="application/zip",
                    use_container_width=True
                )
                st.success("ZIP file ready for download.")

            except Exception as zip_all_error:
                st.error(f"Error creating comprehensive ZIP file: {zip_all_error}")
                st.expander("Traceback").code(traceback.format_exc())
    else:
        st.info("No data available to create a ZIP file.")


    # --- Q&A Section ---
    if ai_enabled and model: # Check if model was initialized successfully
        st.markdown("---")
        st.header("❓ Ask Questions About This Table")
        st.markdown(f"Uses the extracted text content (`{table_data.get('format', 'TEXT')}` format) for context.")

        # Use the primary extracted content for Q&A context
        table_context_for_qa = table_data.get('content', '')
        # If content is HTML, try to convert DF to CSV for better Q&A context
        if table_data.get('format') == 'HTML' and df is not None:
             try:
                 table_context_for_qa = df.to_csv(index=False)
                 st.caption("Using CSV representation of the table for Q&A.")
             except Exception:
                 st.caption("Using raw HTML content for Q&A.") # Fallback to HTML if CSV fails
        elif table_data.get('format') == 'JSON':
             # Potentially simplify JSON or convert to CSV for better context?
             # For now, just use the JSON string.
             st.caption("Using JSON representation of the table for Q&A.")

        if table_context_for_qa and isinstance(table_context_for_qa, str) and len(table_context_for_qa) > 10: # Basic check for valid context
            # Limit context size to avoid overly large prompts (adjust limit as needed)
            max_context_len = 15000 # Characters
            if len(table_context_for_qa) > max_context_len:
                table_context_for_qa = table_context_for_qa[:max_context_len] + "\n... [Context Truncated]"
                st.warning(f"Table context for Q&A truncated to {max_context_len} characters.", icon="⚠️")

            col1, col2 = st.columns([4, 1])
            with col1:
                question = st.text_input("Your question:", key=f"table_question_{selected_table_index}") # Unique key per table
            with col2:
                submit_question = st.button("Ask Gemini", use_container_width=True, key=f"ask_button_{selected_table_index}")

            if submit_question and question:
                try:
                    with st.spinner("🧠 Thinking..."):
                        # Use the pre-initialized model
                        qa_prompt = f"""You are an assistant designed to answer questions based *only* on the provided table data.

                        **Instructions:**
                        1. Analyze the following table data carefully.
                        2. Answer the user's question based *solely* on the information present in the data.
                        3. If the question can be answered, provide a concise and accurate answer.
                        4. If the question cannot be answered from the provided data (e.g., requires external knowledge, calculation not possible from data, or data is missing), state clearly: "Based on the provided table data, I cannot answer that question."
                        5. If the question is completely unrelated to the table's content, state: "This question does not seem related to the provided table data."
                        6. Do not make up information or perform complex calculations beyond simple lookups or aggregations explicitly supported by the data.

                        **Table Data ({table_data.get('format', 'TEXT')} format):**
                        ```
                        {table_context_for_qa}
                        ```

                        **User Question:** {question}

                        **Answer:**"""

                        response = model.generate_content(qa_prompt, request_options={"timeout": 60})

                        # Simple check for refusal keywords before displaying
                        answer = response.text.strip()
                        refusal_phrases = [
                            "cannot answer that question",
                            "not related to the provided table",
                            "don't have enough information",
                            "cannot fulfill that request",
                            "based on the provided table data", # Often precedes a refusal
                        ]
                        is_refusal = any(phrase in answer.lower() for phrase in refusal_phrases)

                        if is_refusal and len(answer) < 150: # Check length to avoid misclassifying long answers that contain these phrases
                             st.warning(f"💡 {answer}")
                        else:
                             st.success(f"✅ Answer:\n\n{answer}")

                except Exception as e:
                    st.error(f"An error occurred processing your question: {str(e)}")
                    st.expander("See error details").code(traceback.format_exc())
        else:
            st.info("No suitable extracted table data available for Q&A for this selection.")
    elif not ai_enabled:
         st.info("Q&A requires a configured Gemini API Key.")