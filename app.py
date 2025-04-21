# app.py
import streamlit as st
import os
import re
import json
import pandas as pd
from PIL import Image
import google.generativeai as genai
from io import StringIO, BytesIO
import base64
import uuid
import tempfile
import shutil
import zipfile
import requests
# --- Import local modules ---
from getImages import screenshot_tables
from amazonTables import amazon_tables

GEMINI_API_KEY = os.getenv("API_KEY")  # Replace with your actual API key
if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
    st.warning("Gemini API Key not found or not set. Please set it. AI features will be disabled.", icon="⚠️")
    ai_enabled = False
else:
    ai_enabled = True
    genai.configure(api_key=GEMINI_API_KEY) 

# --- Page Setup ---
st.set_page_config(page_title="Web Table Extractor", layout="wide")
st.title("🌐 Web Table Extractor")
st.markdown("Extract data from all tables on webpages. Handles Amazon product comparisons directly, uses AI for others.")

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
        'selected_table_index': 0
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
            
            # Add to session state
            st.session_state.extracted_tables.append(table_data)
            st.session_state.extracted_formats.append(format_type)
            st.session_state.extracted_filenames.append(table_data["filename"])
            st.session_state.data_extracted = True
            
            # Handle any images in the table
            if table_data["has_images"]:
                status.text("Processing images in the Amazon table...")
                
                # Create a temporary directory for images
                with tempfile.TemporaryDirectory() as temp_image_dir:
                    # Process DataFrame to download images and update image paths
                    processed_df = df.copy()
                    image_paths = []  # Track image paths for zip creation
                    
                    for i, row in df.iterrows():
                        for col in df.columns:
                            cell_val = str(row[col])
                            if "<img src=" in cell_val:
                                # Extract image URL
                                img_url_match = re.search(r'src="([^"]+)"', cell_val)
                                if img_url_match:
                                    img_url = img_url_match.group(1)
                                    img_filename = f"{temp_image_dir}/amazon_img_{uuid.uuid4().hex[:8]}.jpg"
                                    
                                    try:
                                        img_response = requests.get(img_url, stream=True, timeout=10)
                                        if img_response.status_code == 200:
                                            with open(img_filename, 'wb') as img_file:
                                                img_file.write(img_response.content)
                                            # Replace image URL with local path in DataFrame
                                            processed_df.at[i, col] = f"Image: {img_filename}"
                                            image_paths.append(img_filename)
                                    except Exception as img_err:
                                        print(f"Error downloading image {img_url}: {img_err}")
                                        processed_df.at[i, col] = f"Image URL: {img_url} (download failed)"
                    
                    # Update the table data with processed dataframe
                    table_data["dataframe_with_local_images"] = processed_df
                    
                    # Update content based on format with local image paths
                    if format_type == "CSV":
                        table_data["content_with_local_images"] = processed_df.to_csv(index=False)
                    elif format_type == "JSON":
                        table_data["content_with_local_images"] = processed_df.to_json(orient="records", indent=2)
                    else:  # HTML
                        # Create HTML with local image paths
                        local_html = processed_df.to_html(escape=False, index=False)
                        table_data["content_with_local_images"] = f"<html><head><meta charset='UTF-8'><title>Amazon Comparison</title></head><body>\n{local_html}\n</body></html>"
                    
                    # Create a ZIP file with all content and images
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        # Add the main content file
                        zipf.writestr(table_data['filename'], table_data['content'])
                        
                        # Add the content with local image references
                        local_filename = table_data['filename'].replace(".", "_with_local_refs.")
                        zipf.writestr(local_filename, table_data['content_with_local_images'])
                        
                        # Add all the images
                        for img_path in image_paths:
                            if os.path.exists(img_path):
                                arcname = os.path.basename(img_path)
                                zipf.write(img_path, f"images/{arcname}")
                    
                    # Store the zip content for download
                    zip_buffer.seek(0)
                    table_data["zip_content"] = zip_buffer.getvalue()
                    table_data["zip_filename"] = "amazon_comparison_with_images.zip"
                
                # After exiting the with block, the temp directory is automatically cleaned up
            
            # Update the table in session state
            st.session_state.extracted_tables[-1] = table_data
            
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
            
            # Create a temporary directory for screenshots
            with tempfile.TemporaryDirectory() as temp_screenshot_dir:
                # Get all tables from the page, using the temp directory for screenshots
                tables_info = screenshot_tables(url, screenshot_dir=temp_screenshot_dir)
                
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
                    
                    # For each table, store screenshot path if available
                    screenshot_path = table_info.get("screenshot", "")
                    if screenshot_path and os.path.exists(screenshot_path):
                        st.session_state.screenshot_filenames.append(screenshot_path)
                    
                    # Process the table data
                    df = table_info.get("dataframe")
                    if df is not None:
                        # Check for images in the dataframe
                        has_images = any("<img src=" in str(cell) for row in df.values for cell in row)
                        
                        # Prepare table data
                        table_data = {
                            "title": table_title,
                            "dataframe": df,
                            "html": table_info.get("html", ""),
                            "source_url": url,
                            "has_images": has_images,
                            "screenshot": screenshot_path
                        }
                        
                        # Handle any images in the table
                        if has_images:
                            # Create a temporary directory for images
                            with tempfile.TemporaryDirectory() as temp_image_dir:
                                # Process DataFrame to download images and update image paths
                                processed_df = df.copy()
                                image_paths = []  # Track image paths for zip creation
                                
                                for i, row in df.iterrows():
                                    for col in df.columns:
                                        cell_val = str(row[col])
                                        if "<img src=" in cell_val:
                                            # Extract image URL
                                            img_url_match = re.search(r'src="([^"]+)"', cell_val)
                                            if img_url_match:
                                                img_url = img_url_match.group(1)
                                                img_filename = f"{temp_image_dir}/{safe_domain}_{safe_title}_img_{uuid.uuid4().hex[:8]}.jpg"
                                                
                                                try:
                                                    img_response = requests.get(img_url, stream=True, timeout=10)
                                                    if img_response.status_code == 200:
                                                        with open(img_filename, 'wb') as img_file:
                                                            img_file.write(img_response.content)
                                                        # Replace image URL with local path in DataFrame
                                                        processed_df.at[i, col] = f"Image: {img_filename}"
                                                        image_paths.append(img_filename)
                                                except Exception as img_err:
                                                    print(f"Error downloading image {img_url}: {img_err}")
                                                    processed_df.at[i, col] = f"Image URL: {img_url} (download failed)"
                                
                                # Update the table data with processed dataframe
                                table_data["dataframe_with_local_images"] = processed_df
                                
                                # Format table data
                                filename_base = f"{safe_domain}_{safe_title}_{extraction_id}"
                                if format_type == "CSV":
                                    table_data["content"] = df.to_csv(index=False)
                                    table_data["format"] = "CSV"
                                    table_data["filename"] = f"{filename_base}.csv"
                                    table_data["content_with_local_images"] = processed_df.to_csv(index=False)
                                
                                elif format_type == "JSON":
                                    table_data["content"] = df.to_json(orient="records", indent=2)
                                    table_data["format"] = "JSON"
                                    table_data["filename"] = f"{filename_base}.json"
                                    table_data["content_with_local_images"] = processed_df.to_json(orient="records", indent=2)
                                
                                else:  # HTML
                                    table_html = table_info.get("html", df.to_html(escape=False, index=False))
                                    table_data["content"] = f"<html><head><meta charset='UTF-8'><title>{table_title}</title></head><body>\n{table_html}\n</body></html>"
                                    table_data["format"] = "HTML"
                                    table_data["filename"] = f"{filename_base}.html"
                                    
                                    # Create HTML with local image paths
                                    local_html = processed_df.to_html(escape=False, index=False)
                                    table_data["content_with_local_images"] = f"<html><head><meta charset='UTF-8'><title>{table_title}</title></head><body>\n{local_html}\n</body></html>"
                                
                                # Create a ZIP file with all content and images
                                zip_buffer = BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                    # Add the main content file
                                    zipf.writestr(table_data['filename'], table_data['content'])
                                    
                                    # Add the content with local image references
                                    local_filename = table_data['filename'].replace(".", "_with_local_refs.")
                                    zipf.writestr(local_filename, table_data['content_with_local_images'])
                                    
                                    # Add all the images
                                    for img_path in image_paths:
                                        if os.path.exists(img_path):
                                            arcname = os.path.basename(img_path)
                                            zipf.write(img_path, f"images/{arcname}")
                                    
                                    # Add screenshot if available
                                    if screenshot_path and os.path.exists(screenshot_path):
                                        arcname = os.path.basename(screenshot_path)
                                        zipf.write(screenshot_path, f"screenshots/{arcname}")
                                
                                # Store the zip content for download
                                zip_buffer.seek(0)
                                table_data["zip_content"] = zip_buffer.getvalue()
                                table_data["zip_filename"] = f"{filename_base}_with_images.zip"
                            
                            # After exiting the with block, the temp directory is automatically cleaned up
                        else:
                            # No images - simpler format for table data
                            filename_base = f"{safe_domain}_{safe_title}_{extraction_id}"
                            if format_type == "CSV":
                                table_data["content"] = df.to_csv(index=False)
                                table_data["format"] = "CSV"
                                table_data["filename"] = f"{filename_base}.csv"
                            elif format_type == "JSON":
                                table_data["content"] = df.to_json(orient="records", indent=2)
                                table_data["format"] = "JSON"
                                table_data["filename"] = f"{filename_base}.json"
                            else:  # HTML
                                table_html = table_info.get("html", df.to_html(escape=False, index=False))
                                table_data["content"] = f"<html><head><meta charset='UTF-8'><title>{table_title}</title></head><body>\n{table_html}\n</body></html>"
                                table_data["format"] = "HTML"
                                table_data["filename"] = f"{filename_base}.html"
                            
                            # Create a ZIP with content and screenshot if available
                            if screenshot_path and os.path.exists(screenshot_path):
                                zip_buffer = BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                    # Add the main content file
                                    zipf.writestr(table_data['filename'], table_data['content'])
                                    
                                    # Add screenshot
                                    arcname = os.path.basename(screenshot_path)
                                    zipf.write(screenshot_path, f"screenshots/{arcname}")
                                
                                # Store the zip content for download
                                zip_buffer.seek(0)
                                table_data["zip_content"] = zip_buffer.getvalue()
                                table_data["zip_filename"] = f"{filename_base}_with_screenshot.zip"
                        
                        # Add to session state
                        st.session_state.extracted_tables.append(table_data)
                        st.session_state.extracted_formats.append(format_type)
                        st.session_state.extracted_filenames.append(table_data["filename"])
                
                # Create a master ZIP with all tables and assets
                if len(st.session_state.extracted_tables) > 0:
                    # Create a master ZIP file with all content
                    master_zip_buffer = BytesIO()
                    with zipfile.ZipFile(master_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        # Add each table's content
                        for idx, table_data in enumerate(st.session_state.extracted_tables):
                            # Add the main content file to a subdirectory with the table index
                            table_dir = f"table_{idx+1}"
                            zipf.writestr(f"{table_dir}/{table_data['filename']}", table_data['content'])
                            
                            # If there's content with local image references, add that too
                            if "content_with_local_images" in table_data:
                                local_filename = table_data['filename'].replace(".", "_with_local_refs.")
                                zipf.writestr(f"{table_dir}/{local_filename}", table_data['content_with_local_images'])
                            
                            # Add screenshot if available
                            screenshot_path = table_data.get("screenshot", "")
                            if screenshot_path and os.path.exists(screenshot_path):
                                arcname = os.path.basename(screenshot_path)
                                zipf.write(screenshot_path, f"{table_dir}/screenshots/{arcname}")
                    
                    # Store the master zip for download
                    master_zip_buffer.seek(0)
                    
                    # Add to session state
                    st.session_state.master_zip_content = master_zip_buffer.getvalue()
                    st.session_state.master_zip_filename = f"{safe_domain}_all_tables_{extraction_id}.zip"
                
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
    if "screenshot" in table_data and table_data["screenshot"] and os.path.exists(table_data["screenshot"]):
        with st.expander("View Table Screenshot", expanded=True):
            try:
                image = Image.open(table_data["screenshot"])
                st.image(image, caption=f"Screenshot: {table_data['title']}", use_container_width=True)
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
    
    # Download links
    st.markdown("---")
    st.subheader("⬇️ Download Options")
    
    col1, col2 = st.columns(2)

    with col1:
        try:
            # Regular content download link
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

    # Center just the ZIP download button
    if "zip_content" in table_data:
        # Create three columns for centering - like a visual container
        left_spacer, center_col, right_spacer = st.columns([1, 2, 1])
        
        # Place the button in the center column
        with center_col:
            st.download_button(
                label=f"Download with Images/Screenshots (ZIP)",
                data=table_data["zip_content"],
                file_name=table_data["zip_filename"],
                mime="application/zip",
                use_container_width=True
            )
    
    
    # Option to download all tables as a zip
    if len(st.session_state.extracted_tables) > 1:
        st.markdown("---")
        
        # If we already have a master ZIP prepared, use it
        if hasattr(st.session_state, 'master_zip_content') and st.session_state.master_zip_content:
            st.download_button(
                label=f"Download All Tables as ZIP",
                data=st.session_state.master_zip_content,
                file_name=st.session_state.master_zip_filename,
                mime="application/zip",
                use_container_width=True
            )
        else:
            # Otherwise, create a new ZIP when requested
            if st.button("Prepare ZIP file with all tables", use_container_width=True):
                try:
                    # Create a temporary directory to gather all files
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Create a BytesIO object to store the zip file
                        zip_buffer = BytesIO()
                        
                        # Create a ZipFile object
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            # Add each table to the zip file
                            for idx, table in enumerate(st.session_state.extracted_tables):
                                table_dir = f"table_{idx+1}"
                                
                                # Add the main content
                                zipf.writestr(f"{table_dir}/{table['filename']}", table['content'])
                                
                                # If there's a version with local image paths, add it too
                                if table.get("has_images") and "content_with_local_images" in table:
                                    local_img_filename = table['filename'].replace(".", "_local_images.")
                                    zipf.writestr(f"{table_dir}/{local_img_filename}", table['content_with_local_images'])
                                
                                # Add screenshot if available
                                screenshot_path = table.get("screenshot", "")
                                if screenshot_path and os.path.exists(screenshot_path):
                                    arcname = os.path.basename(screenshot_path)
                                    zipf.write(screenshot_path, f"{table_dir}/screenshots/{arcname}")
                        
                        # Prepare the download link for the zip file
                        zip_buffer.seek(0)
                        domain_name = re.sub(r'^https?://(www\.)?', '', url).split('/')[0]
                        safe_domain = re.sub(r'[^\w\s-]', '', domain_name).strip().replace('.', '_')
                        zip_filename = f"{safe_domain}_all_tables.zip"
                        
                        st.download_button(
                            label=f"Download All Tables as ZIP",
                            data=zip_buffer,
                            file_name=zip_filename,
                            mime="application/zip",
                            use_container_width=True
                        )
                except Exception as zip_error:
                    st.error(f"Error creating ZIP file: {zip_error}")
    
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
                        model = genai.GenerativeModel('gemini-2.0-flash-lite')
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