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
# --- Import local modules ---
from getImages import screenshot_tables
from amazonTables import amazon_tables

GEMINI_API_KEY = os.getenv("API_KEY") # Replace with your actual API key
if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
    st.warning("Gemini API Key not found or not set. Please set it. AI features will be disabled.", icon="⚠️")
    ai_enabled = False
else:
    ai_enabled = True
    genai.configure(api_key=GEMINI_API_KEY) 

# --- Page Setup ---
st.set_page_config(page_title="Web Table Extractor & Analyzer", layout="wide")
st.title("🌐 Web Table Extractor & Analyzer")
st.markdown("Extract data from tables on webpages. Use AI to extract data directly from table images.")

# Initialize session state variables
if 'extraction_method' not in st.session_state:
    st.session_state.extraction_method = None
if 'data_extracted' not in st.session_state:
    st.session_state.data_extracted = False
if 'extracted_tables' not in st.session_state:
    st.session_state.extracted_tables = []
if 'extracted_formats' not in st.session_state:
    st.session_state.extracted_formats = []
if 'extracted_filenames' not in st.session_state:
    st.session_state.extracted_filenames = []
if 'screenshot_captured' not in st.session_state:
    st.session_state.screenshot_captured = False
if 'screenshot_filenames' not in st.session_state:
    st.session_state.screenshot_filenames = []
if 'selected_table_index' not in st.session_state:
    st.session_state.selected_table_index = 0
if 'downloaded_images' not in st.session_state:
    st.session_state.downloaded_images = []
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'safe_title' not in st.session_state:
    st.session_state.safe_title = "extracted_table"

def get_download_link(content, filename, text, format_type):
    """Creates a download link for text content."""
    if isinstance(content, bytes):
        b64 = base64.b64encode(content).decode()
    else:
        b64 = base64.b64encode(str(content).encode('utf-8')).decode() # Ensure UTF-8
    mime_type = {
        "CSV": "text/csv",
        "JSON": "application/json",
        "HTML": "text/html",
        "TEXT": "text/plain"
    }.get(format_type, "application/octet-stream")
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{text}</a>'
    return href

def extract_image_url(cell_val):
    """Extract image URL from HTML cell value using multiple patterns."""
    # First, try to match standard img tag pattern with various quote styles
    img_url_match = re.search(r'<img[^>]*?src=[\'"]([^\'"]+)[\'"]', cell_val)
    if img_url_match:
        return img_url_match.group(1)
    
    # Try to match data-src or data-a-hires (common in Amazon)
    img_url_match = re.search(r'<img[^>]*?data-src=[\'"]([^\'"]+)[\'"]', cell_val)
    if img_url_match:
        return img_url_match.group(1)
    
    img_url_match = re.search(r'<img[^>]*?data-a-hires=[\'"]([^\'"]+)[\'"]', cell_val)
    if img_url_match:
        return img_url_match.group(1)
    
    # Last resort, try to match any src-like attribute
    img_url_match = re.search(r'<img[^>]*?(?:src|data-[^=]+)=[\'"]([^\'"]+)[\'"]', cell_val)
    if img_url_match:
        return img_url_match.group(1)
    
    return None

def resolve_relative_url(base_url, img_url):
    """Resolve a relative URL to absolute URL."""
    if not img_url:
        return None
    
    # Check if URL is already absolute
    if img_url.startswith(('http://', 'https://', 'data:')):
        return img_url
    
    # Handle data URIs
    if img_url.startswith('data:'):
        return img_url
    
    # Parse base URL
    try:
        parsed_base = urlparse(base_url)
        base_url = f"{parsed_base.scheme}://{parsed_base.netloc}"
        
        # Handle different types of relative URLs
        if img_url.startswith('//'):
            # Protocol-relative URL
            return f"{parsed_base.scheme}:{img_url}"
        elif img_url.startswith('/'):
            # Root-relative URL
            return f"{base_url}{img_url}"
        else:
            # Path-relative URL (more complex, use urljoin)
            return urljoin(base_url, img_url)
    except Exception as e:
        print(f"Error resolving URL {img_url} with base {base_url}: {e}")
        return img_url  # Return original as fallback

def download_image(img_url, img_filename, timeout=15, max_retries=3):
    """Download image with retry logic and better error handling."""
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': urlparse(img_url).scheme + '://' + urlparse(img_url).netloc,
            }
            
            # Handle data URIs
            if img_url.startswith('data:image/'):
                try:
                    # Extract base64 data from data URI
                    data_uri_parts = img_url.split(',', 1)
                    if len(data_uri_parts) > 1 and ';base64,' in data_uri_parts[0]:
                        img_data = base64.b64decode(data_uri_parts[1])
                        with open(img_filename, 'wb') as img_file:
                            img_file.write(img_data)
                        return True
                except Exception as data_uri_err:
                    print(f"Error processing data URI: {data_uri_err}")
                    return False
            
            # Regular URL download
            response = requests.get(img_url, headers=headers, stream=True, timeout=timeout)
            response.raise_for_status()
            
            # Get content type to determine file extension
            content_type = response.headers.get('Content-Type', '')
            
            # Check if we got an image
            if 'image' not in content_type and attempt < max_retries - 1:
                print(f"Warning: URL {img_url} did not return an image (got {content_type}). Retrying...")
                time.sleep(1)
                continue
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(img_filename), exist_ok=True)
            
            # Save the image
            with open(img_filename, 'wb') as img_file:
                for chunk in response.iter_content(chunk_size=8192):
                    img_file.write(chunk)
            
            # Verify file was created and has content
            if os.path.exists(img_filename) and os.path.getsize(img_filename) > 0:
                return True
            else:
                if attempt < max_retries - 1:
                    print(f"Empty image file downloaded for {img_url}. Retrying...")
                    time.sleep(1)
                    continue
                return False
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error downloading image {img_url} (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(1)
            else:
                print(f"Failed to download image after {max_retries} attempts: {img_url}, Error: {e}")
                return False
    
    return False

def process_table_images(table_data, base_url, image_dir="extracted_images"):
    """Process all images in a table, downloading them and updating the DataFrame."""
    if not table_data.get('has_images', False) or 'dataframe' not in table_data:
        return table_data
    
    # Create image directory if it doesn't exist
    os.makedirs(image_dir, exist_ok=True)
    
    # Get a clean filename base
    url_domain = urlparse(base_url).netloc.replace("www.", "")
    safe_domain = re.sub(r'[^\w\s-]', '', url_domain).strip().replace('.', '_')
    safe_title = re.sub(r'[^\w\s-]', '', table_data['title']).strip().replace(' ', '_')
    safe_title = re.sub(r'[-_]+', '_', safe_title)
    
    # Clone the DataFrame
    df = table_data['dataframe'].copy()
    processed = False
    downloaded_images = []
    
    # Process each cell to find and download images
    for i, row in df.iterrows():
        for col in df.columns:
            cell_val = str(row[col])
            if "<img" in cell_val:
                # Extract image URL
                img_url = extract_image_url(cell_val)
                if img_url:
                    # Resolve relative URLs
                    img_url = resolve_relative_url(base_url, img_url)
                    
                    # Generate a unique filename for the image
                    file_ext = os.path.splitext(urlparse(img_url).path)[-1].lower()
                    if not file_ext or file_ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']:
                        file_ext = '.jpg'  # Default extension
                    
                    img_filename = f"{image_dir}/{safe_domain}_{safe_title}_{uuid.uuid4().hex[:8]}{file_ext}"
                    
                    # Download the image
                    if download_image(img_url, img_filename):
                        # Update cell with local image path
                        df.at[i, col] = f"Image: {img_filename}"
                        downloaded_images.append(img_filename)
                        processed = True
                    else:
                        df.at[i, col] = f"Image URL: {img_url} (download failed)"
                        processed = True
                else:
                    df.at[i, col] = f"Image tag found but URL could not be extracted"
                    processed = True
    
    # Update table data with processed DataFrame if changes were made
    if processed:
        table_data['dataframe_with_local_images'] = df
        table_data['downloaded_images'] = downloaded_images
    
    return table_data

def extract_data_from_image(image_path, format_type, table_title="Table"):
    """Extract table data from an image using Gemini."""
    if not ai_enabled:
        return {"status": "error", "message": "AI features are disabled. Please set a valid Gemini API key."}
    
    try:
        # Configure Gemini API (should already be configured if ai_enabled is True)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Open image
        image = Image.open(image_path)
        
        # Create prompt with format specification
        prompt = f"""Extract the table data from this image in {format_type} format.
        
        Be precise and thorough in your extraction. Capture all rows and columns.
        For CSV, use proper comma escaping and quoting.
        For JSON, provide an array of objects where each object represents a row.
        For HTML, create a proper HTML table with <table>, <thead>, <tbody>, <tr>, <th>, and <td> tags.
        
        Only return the extracted data in the requested format without explanations.
        """
        
        # Convert the image for Gemini
        image_part = {"mime_type": f"image/{image.format.lower() if image.format else 'jpeg'}", 
                     "data": image_to_bytes(image)}
        
        # Process with Gemini
        response = model.generate_content([prompt, image_part])
        
        # Get result
        result_text = response.text
        
        # Process based on format type
        if format_type == "JSON":
            # Extract JSON if wrapped in code blocks
            if "```json" in result_text and "```" in result_text:
                json_content = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                json_content = result_text.split("```")[1].split("```")[0].strip()
            else:
                json_content = result_text
            
            try:
                # Parse JSON
                table_data = json.loads(json_content)
                return {
                    "status": "success",
                    "content": json_content,
                    "data": table_data,
                    "filename": f"{table_title.replace(' ', '_')}.json",
                    "format": "JSON"
                }
                
            except json.JSONDecodeError as e:
                return {
                    "status": "error",
                    "message": f"Couldn't parse JSON response: {str(e)}",
                    "content": result_text,
                    "filename": f"{table_title.replace(' ', '_')}.txt",
                    "format": "TEXT"
                }
        
        elif format_type == "CSV":
            # Try to extract CSV content
            if "```csv" in result_text and "```" in result_text:
                csv_content = result_text.split("```csv")[1].split("```")[0].strip()
            elif "```" in result_text:
                csv_content = result_text.split("```")[1].split("```")[0].strip()
            else:
                csv_content = result_text
            
            try:
                # Try to load as dataframe
                df = pd.read_csv(pd.StringIO(csv_content))
                return {
                    "status": "success",
                    "content": csv_content,
                    "data": df,
                    "filename": f"{table_title.replace(' ', '_')}.csv",
                    "format": "CSV"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Couldn't parse CSV response: {str(e)}",
                    "content": result_text,
                    "filename": f"{table_title.replace(' ', '_')}.txt",
                    "format": "TEXT"
                }
        
        elif format_type == "HTML":
            # Try to extract HTML content
            if "```html" in result_text and "```" in result_text:
                html_content = result_text.split("```html")[1].split("```")[0].strip()
            elif "```" in result_text:
                html_content = result_text.split("```")[1].split("```")[0].strip()
            else:
                html_content = result_text
            
            # Basic check if it looks like HTML
            if "<table" in html_content.lower() and "</table>" in html_content.lower():
                return {
                    "status": "success",
                    "content": html_content,
                    "filename": f"{table_title.replace(' ', '_')}.html",
                    "format": "HTML"
                }
            else:
                return {
                    "status": "error",
                    "message": "Response doesn't appear to be valid HTML table",
                    "content": result_text,
                    "filename": f"{table_title.replace(' ', '_')}.txt",
                    "format": "TEXT"
                }
        else:
            return {
                "status": "error",
                "message": f"Unsupported format: {format_type}",
                "content": result_text,
                "filename": f"{table_title.replace(' ', '_')}.txt",
                "format": "TEXT"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error extracting data from image: {str(e)}",
            "format": "TEXT"
        }

def image_to_bytes(image):
    """Convert PIL Image to bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=image.format if image.format else "JPEG")
    return buffer.getvalue()

# --- Input Section ---
url = st.text_input("Enter the URL of the webpage:", key="url_input")
is_amazon_product_page = bool(url and ("amazon.com" in url or "amzn." in url or "amzn.to" in url) and ("/dp/" in url or "/gp/product/" in url))
if is_amazon_product_page:
    st.info("Amazon product page detected. Direct extraction will be attempted.", icon="🛒")
    button_text = "Extract Amazon Tables"
else:
    st.info("General webpage detected. Extracting all tables from the page.", icon="📄")
    button_text = "Extract All Tables"

format_type = st.selectbox(
    "Select desired output format:",
    options=["CSV", "JSON", "HTML"],
    index=0 # Default to CSV
)

# --- Action Button ---
if st.button(button_text, key="main_action_button", disabled=not url):
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
        'downloaded_images': [],
        'extracted_data': None
    })
    
    progress = st.progress(0)
    status = st.empty()
    
    if is_amazon_product_page:
        # --- Amazon Direct Extraction Workflow ---
        st.session_state.extraction_method = 'amazon'
        status.text("Attempting direct extraction from Amazon...")
        progress.progress(25)
        
        # Call the imported function
        extracted_result = amazon_tables(url)
        progress.progress(75)
        
        # Check the status returned by the function
        if extracted_result['status'] == 'success':
            status.text("Amazon table extracted successfully!")
            df = extracted_result["dataframe"]
            html_content = extracted_result["html"]
            
            # Save as a table entry
            table_data = {
                "title": "Amazon Product Comparison",
                "dataframe": df,
                "html": html_content,
                "source_url": url,
                "has_images": any("<img" in str(cell) for row in df.values for cell in row)
            }
            
            # Format table data
            if format_type == "CSV":
                table_data["content"] = df.to_csv(index=False)
                table_data["format"] = "CSV"
                table_data["filename"] = f"amazon_comparison.csv"
            elif format_type == "JSON":
                table_data["content"] = df.to_json(orient="records", indent=2)
                table_data["format"] = "JSON"
                table_data["filename"] = f"amazon_comparison.json"
            else:  # HTML
                table_data["content"] = f"<html><head><meta charset='UTF-8'><title>Amazon Comparison</title></head><body>\n{html_content}\n</body></html>"
                table_data["format"] = "HTML"
                table_data["filename"] = f"amazon_comparison.html"
            
            # Handle any images in the table
            if table_data["has_images"]:
                status.text("Processing images in the Amazon table...")
                # Process images with improved image handling
                processed_table = process_table_images(table_data, url)
                
                # If processing added the dataframe with local images, update content accordingly
                if "dataframe_with_local_images" in processed_table:
                    processed_df = processed_table["dataframe_with_local_images"]
                    
                    # Update content based on format with local image paths
                    if format_type == "CSV":
                        processed_table["content_with_local_images"] = processed_df.to_csv(index=False)
                    elif format_type == "JSON":
                        processed_table["content_with_local_images"] = processed_df.to_json(orient="records", indent=2)
                    else:  # HTML
                        # Create HTML with local image paths
                        local_html = processed_df.to_html(escape=False, index=False)
                        processed_table["content_with_local_images"] = f"<html><head><meta charset='UTF-8'><title>Amazon Comparison</title></head><body>\n{local_html}\n</body></html>"
                
                # Add downloaded images to session state
                if "downloaded_images" in processed_table:
                    st.session_state.downloaded_images.extend(processed_table["downloaded_images"])
                
                # Update the table in session state
                table_data = processed_table
            
            # Add to session state
            st.session_state.extracted_tables.append(table_data)
            st.session_state.extracted_formats.append(format_type)
            st.session_state.extracted_filenames.append(table_data["filename"])
            st.session_state.data_extracted = True
            
            progress.progress(100)
            st.rerun()  # Rerun to show results
        else:
            # Handle different failure statuses from amazon_tables
            status.error(f"Amazon Extraction Failed: {extracted_result.get('message', 'Unknown error')}")
            st.error(f"Details: {extracted_result.get('message', 'No further details.')}")
            progress.progress(100)
            st.session_state.data_extracted = False
    
    else:
        # --- General Webpage Table Extraction Workflow ---
        st.session_state.extraction_method = 'general'
        status.text("Extracting all tables from the webpage...")
        progress.progress(10)
        
        try:
            # Create a unique identifier for this extraction
            extraction_id = uuid.uuid4().hex[:8]
            domain_name = re.sub(r'^https?://(www\.)?', '', url).split('/')[0]
            safe_domain = re.sub(r'[^\w\s-]', '', domain_name).strip().replace('.', '_')
            
            # Get all tables from the page
            tables_info = screenshot_tables(url)
            
            if not tables_info or len(tables_info) == 0:
                status.warning("No tables found on the webpage.")
                progress.progress(100)
                st.warning("No tables were detected on the provided webpage. Try a different URL.")
                st.stop()
            
            status.text(f"Found {len(tables_info)} tables on the webpage. Processing...")
            progress.progress(30)
            
            # Process each table
            for idx, table_info in enumerate(tables_info):
                progress_value = 30 + int(60 * (idx + 1) / len(tables_info))
                progress.progress(progress_value)
                status.text(f"Processing table {idx+1} of {len(tables_info)}...")
                
                table_title = table_info.get("title", f"Table {idx+1}")
                safe_title = re.sub(r'[^\w\s-]', '', table_title).strip().replace(' ', '_')
                
                if not safe_title:
                    safe_title = f"table_{idx+1}"
                
                # For each table, store screenshot if available
                screenshot_path = table_info.get("screenshot", "")
                if screenshot_path and os.path.exists(screenshot_path):
                    st.session_state.screenshot_filenames.append(screenshot_path)
                
                # Process the table data
                df = table_info.get("dataframe")
                if df is not None:
                    # Check for images in the dataframe
                    has_images = any("<img" in str(cell) for row in df.values for cell in row)
                    
                    # Prepare table data
                    table_data = {
                        "title": table_title,
                        "dataframe": df,
                        "html": table_info.get("html", ""),
                        "source_url": url,
                        "has_images": has_images,
                        "screenshot": screenshot_path
                    }
                    
                    # Handle any images in the table with improved image handling
                    if has_images:
                        status.text(f"Processing images in table {idx+1}...")
                        processed_table = process_table_images(table_data, url)
                        
                        # If processing added the dataframe with local images, update table_data
                        if "dataframe_with_local_images" in processed_table:
                            table_data = processed_table
                            
                            # Add downloaded images to session state
                            if "downloaded_images" in processed_table:
                                st.session_state.downloaded_images.extend(processed_table["downloaded_images"])
                    
                    # Format table data
                    filename_base = f"{safe_domain}_{safe_title}_{extraction_id}"
                    if format_type == "CSV":
                        table_data["content"] = df.to_csv(index=False)
                        table_data["format"] = "CSV"
                        table_data["filename"] = f"{filename_base}.csv"
                        
                        if has_images and "dataframe_with_local_images" in table_data:
                            processed_df = table_data["dataframe_with_local_images"]
                            table_data["content_with_local_images"] = processed_df.to_csv(index=False)
                    
                    elif format_type == "JSON":
                        table_data["content"] = df.to_json(orient="records", indent=2)
                        table_data["format"] = "JSON"
                        table_data["filename"] = f"{filename_base}.json"
                        
                        if has_images and "dataframe_with_local_images" in table_data:
                            processed_df = table_data["dataframe_with_local_images"]
                            table_data["content_with_local_images"] = processed_df.to_json(orient="records", indent=2)
                    
                    else:  # HTML
                        table_html = table_info.get("html", df.to_html(escape=False, index=False))
                        table_data["content"] = f"<html><head><meta charset='UTF-8'><title>{table_title}</title></head><body>\n{table_html}\n</body></html>"
                        table_data["format"] = "HTML"
                        table_data["filename"] = f"{filename_base}.html"
                        
                        if has_images and "dataframe_with_local_images" in table_data:
                            processed_df = table_data["dataframe_with_local_images"]
                            local_html = processed_df.to_html(escape=False, index=False)
                            table_data["content_with_local_images"] = f"<html><head><meta charset='UTF-8'><title>{table_title}</title></head><body>\n{local_html}\n</body></html>"
                    
                    # Add to session state
                    st.session_state.extracted_tables.append(table_data)
                    st.session_state.extracted_formats.append(format_type)
                    st.session_state.extracted_filenames.append(table_data["filename"])
            
            # Set flag that data was extracted successfully
            if st.session_state.extracted_tables:
                st.session_state.data_extracted = True
                st.session_state.screenshot_captured = bool(st.session_state.screenshot_filenames)
                status.text(f"Successfully extracted {len(st.session_state.extracted_tables)} tables!")
            else:
                status.warning("No tables could be extracted from the webpage.")
            
            progress.progress(100)
            st.rerun()  # Show results
            
        except Exception as e:
            st.error(f"An error occurred during table extraction: {str(e)}")
            import traceback
            st.expander("Traceback").code(traceback.format_exc())
            progress.progress(100)

# --- Display Results and Download Links ---
if st.session_state.data_extracted and st.session_state.extracted_tables:
    st.markdown("---")
    st.header(f"📊 Extracted Tables ({len(st.session_state.extracted_tables)})")
    
    # Table selection if multiple tables
    if len(st.session_state.extracted_tables) > 1:
        table_titles = [f"{idx+1}. {table['title']}" for idx, table in enumerate(st.session_state.extracted_tables)]
        selected_table_index = st.selectbox(
            "Select a table to view:",
            options=range(len(table_titles)),
            format_func=lambda x: table_titles[x],
            index=st.session_state.selected_table_index
        )
        st.session_state.selected_table_index = selected_table_index
    else:
        selected_table_index = 0
    
    # Get the selected table data
    table_data = st.session_state.extracted_tables[selected_table_index]
    
    # Display table information
    st.subheader(table_data["title"])
    st.caption(f"Source URL: {table_data['source_url']}")
    
    # Display screenshot if available
    screenshot_path = table_data.get("screenshot", "")
    if screenshot_path and os.path.exists(screenshot_path):
        with st.expander("View Table Screenshot", expanded=True):
            try:
                image = Image.open(screenshot_path)
                st.image(image, caption=f"Screenshot: {table_data['title']}", use_container_width=True)
                
                # Add option to extract data from screenshot using Gemini
                if ai_enabled:
                    st.session_state.safe_title = re.sub(r'[^\w\s-]', '', table_data['title']).strip().replace(' ', '_')
                    
                    if st.button("Extract Data from Screenshot with AI", key=f"extract_data_button_{selected_table_index}", use_container_width=True):
                        try:
                            progress = st.progress(0)
                            status = st.empty()
                            status.text("Processing screenshot with Gemini AI...")
                            
                            # Call the extraction function
                            extraction_result = extract_data_from_image(
                                screenshot_path, 
                                format_type, 
                                table_data['title']
                            )
                            
                            progress.progress(50)
                            status.text("Processing table data...")
                            
                            if extraction_result["status"] == "success":
                                # Store the result in session state
                                st.session_state.extracted_data = extraction_result
                                
                                # Display based on format type
                                if format_type == "JSON":
                                    st.subheader("AI-Extracted Data (JSON)")
                                    try:
                                        st.json(extraction_result["data"])
                                    except:
                                        st.code(extraction_result["content"], language="json")
                                
                                elif format_type == "CSV":
                                    st.subheader("AI-Extracted Data (CSV)")
                                    try:
                                        st.dataframe(extraction_result["data"])
                                    except:
                                        st.code(extraction_result["content"], language="text")
                                
                                elif format_type == "HTML":
                                    st.subheader("AI-Extracted Data (HTML)")
                                    st.code(extraction_result["content"], language="html")
                                    
                                    # Also try to render it
                                    st.subheader("Rendered HTML Table")
                                    st.components.v1.html(extraction_result["content"], height=400)
                                
                                # Add download link
                                st.markdown(
                                    get_download_link(
                                        extraction_result["content"],
                                        extraction_result["filename"],
                                        f"Download AI-extracted data as {format_type}",
                                        format_type
                                    ),
                                    unsafe_allow_html=True
                                )
                            else:
                                st.error(f"Error extracting data: {extraction_result.get('message', 'Unknown error')}")
                                
                                # If there's still some content, show it
                                if "content" in extraction_result:
                                    st.text_area("Raw output from Gemini:", extraction_result["content"], height=300)
                            
                            progress.progress(100)
                            status.text("AI extraction completed!")
                            
                        except Exception as e:
                            st.error(f"An error occurred during AI extraction: {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())
            except Exception as img_e:
                st.error(f"Error loading screenshot image: {img_e}")
    
    # Display the dataframe
    df = table_data["dataframe"]
    if df is not None:
        st.dataframe(df)
        st.caption(f"Table Preview ({df.shape[0]} rows, {df.shape[1]} columns)")
        
        # Display raw content
        with st.expander(f"View Raw {table_data['format']} Output"):
            display_format = 'json' if table_data['format'] == 'JSON' else table_data['format'].lower()
            try:
                if table_data['format'] == "HTML": 
                    st.code(table_data['content'], language='html')
                elif table_data['format'] == "JSON": 
                    st.json(table_data['content'])
                else: 
                    st.text(table_data['content'])
            except Exception as display_err:
                st.text(f"Raw content:\n{table_data['content']}")
        
        # If table has images that were processed with local paths
        if table_data.get("has_images") and "dataframe_with_local_images" in table_data:
            with st.expander("View Table With Local Image Paths"):
                st.dataframe(table_data["dataframe_with_local_images"])
                st.caption("Table with local file paths to downloaded images")
                
                # Show downloaded images preview
                if "downloaded_images" in table_data and table_data["downloaded_images"]:
                    st.subheader("Downloaded Images")
                    image_cols = st.columns(min(3, len(table_data["downloaded_images"])))
                    for i, img_path in enumerate(table_data["downloaded_images"][:9]):  # Limit to 9 images for preview
                        try:
                            if os.path.exists(img_path):
                                with image_cols[i % 3]:
                                    img = Image.open(img_path)
                                    st.image(img, caption=os.path.basename(img_path), width=150)
                        except Exception as img_err:
                            print(f"Error displaying image {img_path}: {img_err}")
    
    # Download links
    st.markdown("---")
    st.subheader("⬇️ Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
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
    
    # Option to download version with local image paths if available
    if table_data.get("has_images") and "content_with_local_images" in table_data:
        with col2:
            try:
                local_img_filename = table_data['filename'].replace(".", "_local_images.")
                download_link = get_download_link(
                    table_data['content_with_local_images'],
                    local_img_filename,
                    f"Download with local image paths ({local_img_filename})",
                    table_data['format']
                )
                st.markdown(download_link, unsafe_allow_html=True)
            except Exception as dl_error:
                st.error(f"Error generating local images download link: {dl_error}")
    
    # Option to download images for the current table
    if table_data.get("has_images") and "downloaded_images" in table_data and table_data["downloaded_images"]:
        st.markdown("---")
        st.subheader("📸 Download Images")
        
        # Create a zip file with just the current table's images
        if st.button("Prepare ZIP file with images", key="download_images_button"):
            try:
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for img_path in table_data["downloaded_images"]:
                        if os.path.exists(img_path):
                            zipf.write(img_path, os.path.basename(img_path))
                
                zip_buffer.seek(0)
                table_name = re.sub(r'[^\w\s-]', '', table_data['title']).strip().replace(' ', '_')
                zip_filename = f"{table_name}_images.zip"
                
                st.download_button(
                    label=f"Download {len(table_data['downloaded_images'])} Images as ZIP",
                    data=zip_buffer,
                    file_name=zip_filename,
                    mime="application/zip",
                    use_container_width=True
                )
                
            except Exception as zip_error:
                st.error(f"Error creating image ZIP file: {zip_error}")
    
    # Option to download all tables as a zip with images
    if len(st.session_state.extracted_tables) > 1:
        st.markdown("---")
        if st.button("Prepare ZIP file with all tables and images", use_container_width=True):
            try:
                import zipfile
                from io import BytesIO
                
                # Create a BytesIO object to store the zip file
                zip_buffer = BytesIO()
                
                # Create a ZipFile object
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add each table to the zip file
                    for idx, table in enumerate(st.session_state.extracted_tables):
                        filename = table['filename']
                        content = table['content']
                        
                        # Add the content to the zip file
                        zipf.writestr(filename, content)
                        
                        # If there's a version with local image paths, add it too
                        if table.get("has_images") and "content_with_local_images" in table:
                            local_img_filename = filename.replace(".", "_local_images.")
                            zipf.writestr(local_img_filename, table['content_with_local_images'])
                        
                        # Add any screenshots for this table
                        if "screenshot" in table and table["screenshot"] and os.path.exists(table["screenshot"]):
                            screenshot_path = table["screenshot"]
                            zipf.write(screenshot_path, f"screenshots/{os.path.basename(screenshot_path)}")
                        
                        # Add all downloaded images for this table
                        if "downloaded_images" in table and table["downloaded_images"]:
                            for img_path in table["downloaded_images"]:
                                if os.path.exists(img_path):
                                    zipf.write(img_path, f"images/{os.path.basename(img_path)}")
                
                # Prepare the download link for the zip file
                zip_buffer.seek(0)
                domain_name = re.sub(r'^https?://(www\.)?', '', url).split('/')[0]
                safe_domain = re.sub(r'[^\w\s-]', '', domain_name).strip().replace('.', '_')
                zip_filename = f"{safe_domain}_all_tables_with_images.zip"
                
                st.download_button(
                    label=f"Download All Tables and Images as ZIP",
                    data=zip_buffer,
                    file_name=zip_filename,
                    mime="application/zip",
                    use_container_width=True
                )
            except Exception as zip_error:
                st.error(f"Error creating ZIP file: {zip_error}")
                with st.expander("See error details"):
                    st.exception(zip_error)
    
    # --- Q&A Section ---
    if ai_enabled:
        st.markdown("---")
        st.header("❓ Ask Questions About Your Table")
        st.markdown("Uses the extracted text content for context.")
        
        table_context_for_qa = table_data['content']
        if table_context_for_qa:
            col1, col2 = st.columns([4, 1])
            with col1:
                question = st.text_input("Your question:", key="table_question_input")
            with col2:
                submit_question = st.button("Ask Gemini", use_container_width=True, key="ask_button")
            
            if submit_question and question:
                try:
                    with st.spinner("🧠 Thinking..."):
                        # Model should already be configured if ai_enabled is True
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        qa_prompt = f"""Based *only* on the following table data, answer the user's question accurately. If the question cannot be answered from the provided data, state that clearly. Do not make up information.
                        If the question is unrelated to the table or extracted data reply with "UNRELATED: Question unrelated to table, Please ask a question related to the table".
                        Table Data ({table_data['format']} format):
                        ```
                        {table_context_for_qa}
                        ```
                        User Question: {question}
                        Answer:"""
                        
                        response = model.generate_content(qa_prompt, request_options={"timeout": 60})
                        answer = response.text.strip()
                        
                        if answer.startswith("UNRELATED:"):
                            st.warning(answer.replace("UNRELATED: ", ""))
                        else:
                            st.success(answer)
                except Exception as e:
                    st.error(f"An error occurred processing your question: {str(e)}")
                    with st.expander("See error details"):
                        st.exception(e)
        else:
            st.info("No extracted data available to ask questions about.")

# Add option to upload and extract data from an image
st.markdown("---")
st.header("📷 Extract Data from Table Image")
st.markdown("Upload an image containing a table and extract structured data using AI.")

uploaded_file = st.file_uploader("Upload an image with a table", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Save the uploaded image temporarily
    img_path = f"uploaded_table_{uuid.uuid4().hex[:8]}.jpg"
    image.save(img_path)
    
    # Update session state
    st.session_state.safe_title = "uploaded_table"
    
    # Format selection for extraction
    extract_format = st.selectbox(
        "Select output format for extraction:",
        options=["CSV", "JSON", "HTML"],
        index=0,
        key="extract_format"
    )
    
    # Extract button
    if st.button("Extract Data from Image", use_container_width=True, key="extract_uploaded_button"):
        if not ai_enabled:
            st.error("AI features are disabled. Please set a valid Gemini API key.")
        else:
            try:
                with st.spinner("Extracting data from image with Gemini AI..."):
                    # Call the extraction function
                    extraction_result = extract_data_from_image(
                        img_path, 
                        extract_format, 
                        "Uploaded_Table"
                    )
                    
                    if extraction_result["status"] == "success":
                        # Store the result in session state
                        st.session_state.extracted_data = extraction_result
                        
                        st.success("Data extracted successfully!")
                        
                        # Display based on format type
                        if extract_format == "JSON":
                            st.subheader("Extracted Data (JSON)")
                            try:
                                st.json(extraction_result["data"])
                            except:
                                st.code(extraction_result["content"], language="json")
                        
                        elif extract_format == "CSV":
                            st.subheader("Extracted Data (CSV)")
                            try:
                                st.dataframe(extraction_result["data"])
                            except:
                                st.code(extraction_result["content"], language="text")
                        
                        elif extract_format == "HTML":
                            st.subheader("Extracted Data (HTML)")
                            st.code(extraction_result["content"], language="html")
                            
                            # Also try to render it
                            st.subheader("Rendered HTML Table")
                            st.components.v1.html(extraction_result["content"], height=400)
                        
                        # Add download link
                        st.markdown(
                            get_download_link(
                                extraction_result["content"],
                                extraction_result["filename"],
                                f"Download extracted data as {extract_format}",
                                extract_format
                            ),
                            unsafe_allow_html=True
                        )
                    else:
                        st.error(f"Error extracting data: {extraction_result.get('message', 'Unknown error')}")
                        
                        # If there's still some content, show it
                        if "content" in extraction_result:
                            st.text_area("Raw output from Gemini:", extraction_result["content"], height=300)
            
            except Exception as e:
                st.error(f"An error occurred during extraction: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
            
            # Clean up temporary file
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
            except:
                pass