# amazonTables.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re # For regex selector
import os
import uuid
from io import StringIO # Potentially needed if pandas uses it internally

def amazon_tables(url):
    """
    Scrapes all tables from an Amazon product page, including the product comparison table.
    Args:
        url (str): The URL of the Amazon product page.
    Returns:
        dict: A dictionary containing:
              - 'status': 'success', 'not_found', 'request_error', 'parse_error'
              - 'dataframe': pandas DataFrame (if status is 'success')
              - 'html': HTML string of the table (if status is 'success')
              - 'message': Error or status message (if status is not 'success')
              - 'tables': List of all tables found (if multiple tables are present)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Upgrade-Insecure-Requests': '1',
        'Connection': 'keep-alive',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache',
    }
    
    print(f"Requesting Amazon URL: {url}") # Keep print for console feedback
    
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        print(f"Response Status Code: {response.status_code}")
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Track all tables found
        all_tables = []
        primary_table = None
        
        # 1. Look for the comparison table first (most common and typically the main table)
        comparison_div = soup.find("div", id="comparison-chart")
        comparison_table = comparison_div.find("table") if comparison_div else None
        
        if not comparison_table:
            comparison_table = soup.find("table", class_=re.compile(r'_product-comparison-.*comparisonTable'))
        
        if comparison_table:
            print("Comparison table found in HTML.")
            primary_table = extract_amazon_table(comparison_table, "Product Comparison")
            if primary_table:
                all_tables.append(primary_table)
        
        # 2. Look for specification tables (commonly found in product pages)
        spec_tables_selectors = [
            "#productDetails table", 
            "#technicalSpecifications_section_1 table",
            "#tech-specs table",
            "#technical-data table",
            ".a-bordered table",
            "#detailBullets_feature_div",
            "#technical-details table",
            "#productDetails_techSpec_section_1"
        ]
        
        for selector in spec_tables_selectors:
            try:
                spec_tables = soup.select(selector)
                for idx, spec_table in enumerate(spec_tables):
                    table_title = f"Specifications Table {idx+1}"
                    
                    # Try to find a better title
                    try:
                        # Look for a header near the table
                        parent = spec_table.parent
                        for _ in range(3):  # Look up to 3 levels
                            if parent:
                                headers = parent.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
                                if headers:
                                    table_title = headers[0].get_text(strip=True)
                                    break
                                parent = parent.parent
                    except:
                        pass
                    
                    table_data = extract_amazon_table(spec_table, table_title)
                    if table_data:
                        all_tables.append(table_data)
            except Exception as e:
                print(f"Error finding specification tables with selector {selector}: {e}")
        
        # 3. Look for any other tables in the page
        all_html_tables = soup.find_all("table")
        for idx, table in enumerate(all_html_tables):
            # Skip tables we've already processed
            if comparison_table and table == comparison_table:
                continue
            
            if any(table == t.get('element') for t in all_tables if 'element' in t):
                continue
            
            table_title = f"Amazon Table {idx+1}"
            
            # Try to find a better title
            try:
                # Check for caption
                caption = table.find("caption")
                if caption:
                    table_title = caption.get_text(strip=True)
                else:
                    # Look for a header near the table
                    parent = table.parent
                    for _ in range(3):  # Look up to 3 levels
                        if parent:
                            headers = parent.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
                            if headers:
                                table_title = headers[0].get_text(strip=True)
                                break
                            parent = parent.parent
            except:
                pass
                
            table_data = extract_amazon_table(table, table_title)
            if table_data:
                all_tables.append(table_data)
        
        # 4. Look for the "detailBulletsWrapper_feature_div" which often contains product details in a list format
        detail_bullets = soup.find(id="detailBulletsWrapper_feature_div")
        if detail_bullets:
            try:
                # Convert the bullet points to a table format
                bullet_points = detail_bullets.select("li")
                if bullet_points:
                    rows = []
                    for li in bullet_points:
                        text = li.get_text(strip=True)
                        # Split on common separators
                        parts = re.split(r':\s*|\s*‎\s*|\s*•\s*', text, 1)
                        if len(parts) == 2:
                            rows.append(parts)
                        elif len(parts) == 1 and text:
                            rows.append([f"Item {len(rows)+1}", text])
                    
                    if rows:
                        df = pd.DataFrame(rows, columns=["Specification", "Value"])
                        html_table = df.to_html(escape=False, index=False, border=1, classes='dataframe table table-striped table-hover')
                        
                        bullet_table = {
                            'title': "Product Details",
                            'dataframe': df,
                            'html': html_table
                        }
                        all_tables.append(bullet_table)
            except Exception as e:
                print(f"Error processing detail bullets: {e}")
        
        # Prepare the response
        if not all_tables:
            print("No tables found on the Amazon page using known selectors.")
            return {'status': 'not_found', 'message': "No tables found on the Amazon page."}
        
        # Handle images if present in any tables
        for table in all_tables:
            if 'dataframe' in table:
                df = table['dataframe']
                has_images = any("<img" in str(cell) for row in df.values for cell in row)
                
                if has_images:
                    # Process image URLs to local files if needed in the future
                    table['has_images'] = True
        
        # Set the primary table if we don't have one yet
        if not primary_table and all_tables:
            primary_table = all_tables[0]
        
        # Return success with tables
        result = {
            'status': 'success', 
            'dataframe': primary_table['dataframe'],
            'html': primary_table['html'],
            'tables': all_tables
        }
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch the Amazon page. Error: {e}")
        return {'status': 'request_error', 'message': f"Failed to fetch the Amazon page. Error: {e}"}
    
    except Exception as e:
        print(f"An unexpected error occurred during Amazon table extraction: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'parse_error', 'message': f"An unexpected error occurred during Amazon table extraction: {e}"}

def extract_amazon_table(table_element, table_title):
    """
    Extract data from an Amazon table element into a structured format
    Args:
        table_element: BeautifulSoup table element
        table_title: Title to use for the table
    Returns:
        dict: Table data including DataFrame and HTML, or None if extraction failed
    """
    try:
        rows_data = []
        table_rows = table_element.find_all("tr")
        
        if not table_rows:
            print(f"Warning: Table '{table_title}' contains no 'tr' (row) elements.")
            return None
        
        print(f"Found {len(table_rows)} 'tr' elements in table '{table_title}'.")
        
        for tr in table_rows:
            row = []
            for cell in tr.find_all(["td", "th"]):
                img = cell.find("img")
                if img:
                    image_url = img.get("data-a-hires") or img.get("src") or ""
                    row.append(f'<img src="{image_url}" style="max-width:50px; max-height:50px;">')
                else:
                    cell_text = ' '.join(cell.get_text(strip=True).split())
                    row.append(cell_text)
            
            if row and any(c for c in row if str(c).strip()):
                rows_data.append(row)
        
        if not rows_data:
            print(f"Warning: Table '{table_title}' found, but no data rows could be extracted.")
            return None
        
        headers_list = []
        header_row_element = table_element.find("thead")
        
        if header_row_element:
            header_cells = header_row_element.find_all("th")
            if header_cells:
                headers_list = [' '.join(th.get_text(strip=True).split()) for th in header_cells]
                print(f"Extracted headers from thead: {headers_list}")
        
        df = None
        if rows_data:
            if not headers_list:
                first_row_cells = rows_data[0]
                if not any('<img' in str(cell) for cell in first_row_cells) and any(isinstance(cell, str) and cell for cell in first_row_cells):
                    print(f"Assuming first extracted row is the header for table '{table_title}'.")
                    headers_list = first_row_cells
                    rows_data = rows_data[1:]
            
            if rows_data:
                if headers_list and len(headers_list) == len(rows_data[0]):
                    df = pd.DataFrame(rows_data, columns=headers_list)
                else:
                    print(f"Header count mismatch or headers missing for table '{table_title}'. Creating DF with generic headers.")
                    df = pd.DataFrame(rows_data)
                    df.columns = [f"Column {i+1}" for i in range(df.shape[1])]
        
        if df is not None and not df.empty:
            print(f"DataFrame created successfully for table '{table_title}'.")
            html_table = df.to_html(escape=False, index=False, border=1, classes='dataframe table table-striped table-hover')
            
            return {
                'title': table_title,
                'dataframe': df,
                'html': html_table,
                'element': table_element  # Store reference to original element to avoid duplicates
            }
        else:
            print(f"Could not create a valid DataFrame from table '{table_title}'.")
            return None
    
    except Exception as e:
        print(f"Error extracting data from table '{table_title}': {e}")
        return None

# For testing purposes
if __name__ == "__main__":
    # Example Amazon product URL
    url = "https://www.amazon.com/dp/B09H1LZ21F"  # Replace with an actual Amazon product URL
    result = amazon_tables(url)
    
    if result['status'] == 'success':
        print("\nExtracted Tables:")
        for idx, table in enumerate(result['tables']):
            print(f"\nTable {idx+1}: {table['title']}")
            print(f"Columns: {list(table['dataframe'].columns)}")
            print(f"Rows: {len(table['dataframe'])}")
            print(f"Sample data: {table['dataframe'].head(2)}")
    else:
        print(f"Extraction failed: {result['message']}")