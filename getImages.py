import sys
import time
import re
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import uuid
from bs4 import BeautifulSoup
import requests

def accept_cookies(driver, max_attempts=3):
    """
    Detect and accept common cookie consent banners and overlays
    Args:
        driver: Selenium WebDriver instance
        max_attempts: Maximum number of attempts to find and click consent buttons
    """
    print("Checking for cookie consent banners...")
    # Common terms found in cookie acceptance buttons and their containing elements
    consent_button_patterns = [
        # Button text patterns (case insensitive)
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]",
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree')]",
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'consent')]",
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'got it')]",
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'i agree')]",
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'ok')]",
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'allow')]",
        "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue')]",
        # Links or anchor tags
        "//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]",
        "//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree')]",
        "//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'consent')]",
        # Input buttons
        "//input[@type='button' and contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]",
        "//input[@type='button' and contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree')]",
        # Div and span elements acting as buttons
        "//div[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept') and (@role='button' or contains(@class, 'btn') or contains(@class, 'button'))]",
        "//div[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree') and (@role='button' or contains(@class, 'btn') or contains(@class, 'button'))]",
        "//span[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept') and (@role='button' or contains(@class, 'btn') or contains(@class, 'button'))]",
        # IDs and classes
        "//*[contains(@id, 'cookie-accept') or contains(@id, 'accept-cookie') or contains(@id, 'cookieAccept')]",
        "//*[contains(@class, 'cookie-accept') or contains(@class, 'accept-cookie') or contains(@class, 'cookieAccept')]",
        "//*[contains(@id, 'cookie-agree') or contains(@id, 'agree-cookie') or contains(@id, 'cookieAgree')]",
        "//*[contains(@id, 'cookie-consent') or contains(@id, 'consent-cookie') or contains(@id, 'cookieConsent')]"
    ]
    # Try each attempt
    for attempt in range(max_attempts):
        for xpath in consent_button_patterns:
            try:
                # Short wait to find the element
                buttons = driver.find_elements(By.XPATH, xpath)
                for button in buttons:
                    # Check if element is visible
                    if button.is_displayed():
                        print(f"Found consent button: {button.text or button.get_attribute('value') or button.get_attribute('id') or 'unnamed button'}")
                        button.click()
                        print("Clicked consent button")
                        time.sleep(1)  # Wait for overlay to disappear
                        return True
            except Exception as e:
                # Just continue to the next pattern
                pass
        # If no button found on this attempt, wait a bit and try again
        if attempt < max_attempts - 1:
            time.sleep(1)
    
    # Special cases for iframes (some consent banners are in iframes)
    try:
        frames = driver.find_elements(By.TAG_NAME, "iframe")
        for frame in frames:
            try:
                # Check if the frame might be a cookie consent frame
                frame_id = frame.get_attribute("id") or ""
                frame_name = frame.get_attribute("name") or ""
                if ("cookie" in frame_id.lower() or
                    "consent" in frame_id.lower() or
                    "cookie" in frame_name.lower() or
                    "consent" in frame_name.lower()):
                    # Switch to this frame
                    driver.switch_to.frame(frame)
                    print(f"Switched to frame: {frame_id or frame_name}")
                    # Try all our patterns in this frame
                    for xpath in consent_button_patterns:
                        try:
                            buttons = driver.find_elements(By.XPATH, xpath)
                            for button in buttons:
                                if button.is_displayed():
                                    button.click()
                                    print(f"Clicked consent button in iframe: {button.text or 'unnamed button'}")
                                    time.sleep(1)
                                    driver.switch_to.default_content()
                                    return True
                        except:
                            pass
                    # Switch back to main content
                    driver.switch_to.default_content()
            except:
                driver.switch_to.default_content()
                continue
    except Exception as e:
        print(f"Error checking frames: {e}")
    
    print("No cookie consent buttons found or unable to interact with them")
    return False

def extract_table_data(table_element, driver):
    """
    Extract data from a table element into a pandas DataFrame
    Args:
        table_element: Selenium WebElement representing a table
        driver: Selenium WebDriver instance
    Returns:
        dict: A dictionary containing the DataFrame, HTML, and other table info
    """
    try:
        # Get table HTML
        table_html = table_element.get_attribute('outerHTML')
        
        # Get table caption or title
        table_title = ""
        try:
            # Try to find caption within the table
            caption = table_element.find_element(By.TAG_NAME, "caption")
            if caption:
                table_title = caption.text.strip()
        except NoSuchElementException:
            # Try to find a nearby heading
            try:
                # Look for headings that could be associated with this table
                script = """
                function findTableHeading(table) {
                    // Check previous siblings for headings
                    let el = table;
                    while (el = el.previousElementSibling) {
                        if (el.tagName.match(/^H[1-6]$/)) {
                            return el.textContent.trim();
                        }
                    }
                    
                    // Check parent for heading
                    let parent = table.parentElement;
                    if (parent) {
                        // Check for heading within the parent before the table
                        let found = false;
                        for (let i = 0; i < parent.children.length; i++) {
                            let child = parent.children[i];
                            if (child === table) {
                                found = true;
                                break;
                            }
                            if (child.tagName.match(/^H[1-6]$/)) {
                                return child.textContent.trim();
                            }
                        }
                    }
                    
                    // Check parent's parent for section title
                    let grandparent = parent ? parent.parentElement : null;
                    if (grandparent) {
                        let headings = grandparent.querySelectorAll('h1, h2, h3, h4, h5, h6');
                        if (headings.length > 0) {
                            return headings[headings.length - 1].textContent.trim();
                        }
                    }
                    
                    return "";
                }
                return findTableHeading(arguments[0]);
                """
                table_title = driver.execute_script(script, table_element)
            except Exception as title_err:
                print(f"Error finding table title: {title_err}")
                
        # If no title found, use generic name
        if not table_title:
            # Try to infer from table classes or ID
            table_id = table_element.get_attribute('id') or ""
            table_class = table_element.get_attribute('class') or ""
            
            if table_id:
                table_title = f"Table: {table_id}"
            elif table_class:
                table_title = f"Table: {table_class.split()[0]}"
            else:
                table_title = "Untitled Table"
        
        # Use BeautifulSoup to parse the HTML table
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(table_html, 'html.parser')
        
        # Extract headers (prioritize thead > tr > th)
        headers = []
        thead = soup.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
        
        # If no headers in thead, try first row
        if not headers:
            first_row = soup.find('tr')
            if first_row:
                for cell in first_row.find_all(['th', 'td']):
                    # Check if it looks like a header (e.g., has bold text or is a th)
                    if cell.name == 'th' or cell.find('strong') or cell.find('b'):
                        headers.append(cell.get_text(strip=True))
                    else:
                        headers.append(cell.get_text(strip=True))
        
        # If still no headers, create generic column names
        if not headers and soup.find_all('tr'):
            first_row = soup.find('tr')
            num_cols = len(first_row.find_all(['th', 'td']))
            headers = [f"Column {i+1}" for i in range(num_cols)]
        
        # Extract rows
        rows = []
        for tr in soup.find_all('tr'):
            # Skip header row if we already processed it
            if tr.parent and tr.parent.name == 'thead' and headers:
                continue
            
            # Skip if it's the first row and we used it for headers
            if not rows and tr == soup.find('tr') and len(headers) > 0:
                # Check if this is likely a header row
                cells = tr.find_all(['th', 'td'])
                header_like = any(cell.name == 'th' for cell in cells)
                if header_like:
                    continue
            
            row = []
            for cell in tr.find_all(['td', 'th']):
                # Check for images and preserve them
                img = cell.find('img')
                if img:
                    img_src = img.get('src', '')
                    if img_src:
                        # Create HTML image tag to preserve in the data
                        row.append(f'<img src="{img_src}" style="max-width:50px; max-height:50px;">')
                    else:
                        row.append(cell.get_text(strip=True))
                else:
                    row.append(cell.get_text(strip=True))
            
            # Only add non-empty rows
            if any(cell.strip() for cell in row if isinstance(cell, str)):
                rows.append(row)
        
        # Create DataFrame
        df = None
        if rows and headers and len(rows[0]) == len(headers):
            df = pd.DataFrame(rows, columns=headers)
        elif rows:
            # If header count doesn't match row cell count, create generic headers
            max_cols = max(len(row) for row in rows)
            generic_headers = [f"Column {i+1}" for i in range(max_cols)]
            # Pad rows with empty strings if needed
            padded_rows = [row + [''] * (max_cols - len(row)) for row in rows]
            df = pd.DataFrame(padded_rows, columns=generic_headers)
        
        # If we have a dataframe, return it along with other info
        if df is not None and not df.empty:
            return {
                'title': table_title,
                'dataframe': df,
                'html': table_html
            }
        else:
            return None
    
    except Exception as e:
        print(f"Error extracting table data: {e}")
        return None

def get_div_table_data(div_element, driver):
    """
    Extract data from a div that looks like a table
    Args:
        div_element: Selenium WebElement representing a div-based table
        driver: Selenium WebDriver instance
    Returns:
        dict: A dictionary containing the DataFrame, HTML, and other table info
    """
    try:
        # Get div HTML
        div_html = div_element.get_attribute('outerHTML')
        
        # Try to determine if this is a table-like structure
        class_name = div_element.get_attribute('class') or ''
        table_like_classes = ['table', 'grid', 'data-table', 'datatable', 'standings']
        
        is_table_like = any(cls in class_name.lower() for cls in table_like_classes)
        
        if not is_table_like:
            # Check for regular structure indicating a table
            row_elements = div_element.find_elements(By.XPATH, './div[contains(@class, "row") or contains(@class, "tr")]')
            if len(row_elements) < 2:  # Need at least a header row and one data row
                row_elements = div_element.find_elements(By.XPATH, './ul | ./ol')
                
            if len(row_elements) < 2:
                # Not enough row structure to be a table
                return None
        
        # Get table title
        table_title = ""
        try:
            # Look for headings near this div
            script = """
            function findDivTableHeading(div) {
                // Check for a heading just before this div
                let el = div;
                while (el = el.previousElementSibling) {
                    if (el.tagName.match(/^H[1-6]$/)) {
                        return el.textContent.trim();
                    }
                }
                
                // Check parent's children before this div
                let parent = div.parentElement;
                if (parent) {
                    let found = false;
                    for (let i = 0; i < parent.children.length; i++) {
                        let child = parent.children[i];
                        if (child === div) {
                            found = true;
                            break;
                        }
                        if (child.tagName.match(/^H[1-6]$/)) {
                            return child.textContent.trim();
                        }
                    }
                }
                
                // Look for a title/caption div within the table-like div
                let titleElements = div.querySelectorAll('.title, .caption, .header, .heading');
                if (titleElements.length > 0) {
                    return titleElements[0].textContent.trim();
                }
                
                return "";
            }
            return findDivTableHeading(arguments[0]);
            """
            table_title = driver.execute_script(script, div_element)
        except Exception as title_err:
            print(f"Error finding div table title: {title_err}")
        
        # If no title found, use generic name
        if not table_title:
            div_id = div_element.get_attribute('id') or ""
            if div_id:
                table_title = f"Table: {div_id}"
            else:
                table_title = f"Table: {class_name}" if class_name else "Untitled Table"
        
        # Use BeautifulSoup to parse the div structure
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(div_html, 'html.parser')
        
        # Try different strategies to extract header and rows
        
        # Strategy 1: Look for explicit row/column classes
        headers = []
        rows = []
        
        # Look for header elements
        header_elements = soup.select('.header, .heading, .headers, th, [role="columnheader"]')
        if header_elements:
            headers = [h.get_text(strip=True) for h in header_elements]
        
        # Look for row elements
        row_elements = soup.select('.row, .data-row, tr, li, [role="row"]')
        for row_el in row_elements:
            # Skip if this looks like a header row and we already have headers
            if (row_el.select_one('.header, .heading, th') or 'header' in row_el.get('class', [])) and headers:
                continue
                
            # Get cells
            cell_elements = row_el.select('.cell, .column, td, span')
            if not cell_elements:
                # Try direct children if no specific cell classes
                cell_elements = [child for child in row_el.children if child.name]
                
            row = []
            for cell in cell_elements:
                # Check for images
                img = cell.find('img')
                if img:
                    img_src = img.get('src', '')
                    if img_src:
                        row.append(f'<img src="{img_src}" style="max-width:50px; max-height:50px;">')
                    else:
                        row.append(cell.get_text(strip=True))
                else:
                    row.append(cell.get_text(strip=True))
                    
            # Only add non-empty rows
            if row and any(str(cell).strip() for cell in row):
                rows.append(row)
        
        # If that didn't work, try a simpler approach looking for regular structure
        if not rows:
            # Assume first row might be header
            first_row = soup.find(class_=lambda c: c and ('row' in c or 'header' in c))
            
            # If no classes help, look for consistent structure
            if not first_row:
                # Look for sets of similar elements that might be rows
                direct_children = [c for c in soup.find('div').children if c.name]
                
                # Group by tag name to find the most common tag that could represent rows
                from collections import Counter
                tag_counts = Counter(child.name for child in direct_children if child.name)
                
                if tag_counts:
                    most_common_tag = tag_counts.most_common(1)[0][0]
                    potential_rows = soup.find_all(most_common_tag, recursive=False)
                    
                    if potential_rows:
                        # First row might be header
                        headers = []
                        for cell in potential_rows[0].children:
                            if cell.name:
                                headers.append(cell.get_text(strip=True))
                        
                        # Process remaining rows
                        for row_el in potential_rows[1:]:
                            row = []
                            for cell in row_el.children:
                                if cell.name:
                                    # Check for images
                                    img = cell.find('img')
                                    if img:
                                        img_src = img.get('src', '')
                                        if img_src:
                                            row.append(f'<img src="{img_src}" style="max-width:50px; max-height:50px;">')
                                        else:
                                            row.append(cell.get_text(strip=True))
                                    else:
                                        row.append(cell.get_text(strip=True))
                            
                            if row and any(str(cell).strip() for cell in row):
                                rows.append(row)
        
        # Create DataFrame
        df = None
        if rows and headers and len(rows[0]) == len(headers):
            df = pd.DataFrame(rows, columns=headers)
        elif rows:
            # If header count doesn't match row cell count or no headers, create generic headers
            max_cols = max(len(row) for row in rows)
            generic_headers = [f"Column {i+1}" for i in range(max_cols)]
            # Pad rows with empty strings if needed
            padded_rows = [row + [''] * (max_cols - len(row)) for row in rows]
            df = pd.DataFrame(padded_rows, columns=generic_headers)
        
        # If we have a dataframe, return it along with other info
        if df is not None and not df.empty:
            return {
                'title': table_title,
                'dataframe': df,
                'html': div_html
            }
        else:
            return None
    
    except Exception as e:
        print(f"Error extracting div table data: {e}")
        return None

def screenshot_table_element(driver, table_element, base_filename, idx=0):
    """
    Take a screenshot of a specific table element
    Args:
        driver: Selenium WebDriver instance
        table_element: WebElement of the table to screenshot
        base_filename: Base filename to use for the screenshot
        idx: Index of the table (for multiple tables)
    Returns:
        str: Path to the screenshot file, or None if failed
    """
    try:
        # Create screenshots directory if it doesn't exist
        os.makedirs('screenshots', exist_ok=True)
        
        # Scroll to the table element to make sure it's visible
        driver.execute_script("arguments[0].scrollIntoView({block: 'center', behavior: 'instant'});", table_element)
        time.sleep(0.5)
        
        # Try to ensure the whole table is visible
        driver.execute_script("""
            // Force table to be fully visible
            arguments[0].style.overflow = 'visible';
            
            // If there are parent containers with overflow:hidden, fix them
            let parent = arguments[0].parentElement;
            for (let i = 0; i < 5 && parent; i++) {
                if (window.getComputedStyle(parent).overflow === 'hidden' ||
                    window.getComputedStyle(parent).overflowX === 'hidden') {
                    parent.style.overflow = 'visible';
                    parent.style.overflowX = 'visible';
                }
                parent = parent.parentElement;
            }
        """, table_element)
        time.sleep(0.5)
        
        # Create screenshot filename
        filename = f"screenshots/{base_filename}_table_{idx}.png"
        
        # Take the screenshot
        table_element.screenshot(filename)
        print(f"Table screenshot saved as {filename}")
        return filename
    
    except Exception as e:
        print(f"Error taking table screenshot: {e}")
        return None

def screenshot_tables(url):
    """
    Find and extract all tables from a webpage
    Args:
        url (str): The URL of the webpage
    Returns:
        list: List of dictionaries containing table data, each with:
             - 'title': Table title/caption
             - 'dataframe': pandas DataFrame of the table data
             - 'html': HTML string of the table
             - 'screenshot': Path to screenshot image (if successful)
    """
    # Set up Chrome options for headless environment
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Initialize WebDriver
    driver = webdriver.Chrome(options=chrome_options)
    
    tables_info = []
    safe_domain = "webpage"
    
    try:
        # Load the webpage
        driver.get(url)
        print(f"Loaded page: {url}")
        
        # Wait for the page to load
        time.sleep(3)
        
        # Handle cookie consent overlays
        accept_cookies(driver)
        
        # Create a safe base filename from the URL
        domain_name = re.sub(r'^https?://(www\.)?', '', url).split('/')[0]
        safe_domain = re.sub(r'[^\w\s-]', '', domain_name).strip().replace('.', '_')
        
        # Find all HTML tables
        print("Looking for HTML tables...")
        tables = driver.find_elements(By.TAG_NAME, "table")
        print(f"Found {len(tables)} HTML tables")
        
        # Process each HTML table
        for idx, table in enumerate(tables):
            try:
                # Check if table is visible and has content
                if not table.is_displayed():
                    print(f"Table {idx+1} is not displayed, skipping")
                    continue
                
                # Extract table data
                table_data = extract_table_data(table, driver)
                
                if table_data:
                    # Take screenshot of the table
                    screenshot = screenshot_table_element(driver, table, safe_domain, idx)
                    if screenshot:
                        table_data['screenshot'] = screenshot
                    
                    tables_info.append(table_data)
                    print(f"Successfully extracted table {idx+1}: {table_data['title']}")
            except Exception as table_err:
                print(f"Error processing table {idx+1}: {table_err}")
        
        # Now look for div-based tables
        print("Looking for div-based tables...")
        div_table_selectors = [
            "div.table", "div.datatable", "div.grid", "div.data-table",
            "div.standings", "div[role='table']", "div.table-responsive",
            "div[class*='table']", "div[class*='grid']"
        ]
        
        for selector in div_table_selectors:
            try:
                div_tables = driver.find_elements(By.CSS_SELECTOR, selector)
                print(f"Found {len(div_tables)} potential div-based tables with selector: {selector}")
                
                for idx, div_table in enumerate(div_tables):
                    try:
                        # Check if this div is visible and likely a table
                        if not div_table.is_displayed():
                            continue
                        
                        # Extract data from div-based table
                        div_table_data = get_div_table_data(div_table, driver)
                        
                        if div_table_data:
                            # Take screenshot
                            screenshot = screenshot_table_element(driver, div_table, f"{safe_domain}_div", len(tables_info))
                            if screenshot:
                                div_table_data['screenshot'] = screenshot
                            
                            tables_info.append(div_table_data)
                            print(f"Successfully extracted div-based table: {div_table_data['title']}")
                    except Exception as div_err:
                        print(f"Error processing div-based table: {div_err}")
            except Exception as selector_err:
                print(f"Error with selector {selector}: {selector_err}")
        
        # If no tables found, try to use AI to extract from the entire page
        if not tables_info:
            print("No standard tables found. Taking a screenshot of the full page.")
            os.makedirs('screenshots', exist_ok=True)
            full_page_file = f"screenshots/{safe_domain}_full_page.png"
            driver.save_screenshot(full_page_file)
            print(f"Full page screenshot saved as {full_page_file}")
            
            # You might want to add AI-based extraction here in the future
        
        # Process table data to handle image URLs
        for table_data in tables_info:
            if 'dataframe' in table_data:
                df = table_data['dataframe']
                # Check if this table has images
                has_images = False
                
                # Process each cell to find image tags
                for i, row in df.iterrows():
                    for col in df.columns:
                        cell_val = str(row[col])
                        if '<img src=' in cell_val:
                            has_images = True
                            break
                    if has_images:
                        break
                
                table_data['has_images'] = has_images
        
        print(f"Total tables extracted: {len(tables_info)}")
        return tables_info
    
    except Exception as e:
        print(f"An error occurred while extracting tables: {e}")
        import traceback
        traceback.print_exc()
        
        # Take screenshot of the entire page as a fallback
        try:
            os.makedirs('screenshots', exist_ok=True)
            driver.save_screenshot(f"screenshots/error_{safe_domain}.png")
            print(f"Error occurred, full page screenshot saved")
        except:
            pass
            
        return []
    
    finally:
        driver.quit()

def download_table_images(table_data, base_dir="extracted_images"):
    """
    Download images found in table data and update DataFrame with local paths
    Args:
        table_data: Dictionary containing table information including DataFrame
        base_dir: Directory to save downloaded images
    Returns:
        Updated table_data with local image paths
    """
    if not table_data.get('has_images', False) or 'dataframe' not in table_data:
        return table_data
    
    # Create image directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Get safe title for filename
    safe_title = re.sub(r'[^\w\s-]', '', table_data['title']).strip().replace(' ', '_')
    safe_title = re.sub(r'[-_]+', '_', safe_title)
    
    # Clone the DataFrame
    df = table_data['dataframe'].copy()
    processed = False
    
    # Process each cell to find and download images
    for i, row in df.iterrows():
        for col in df.columns:
            cell_val = str(row[col])
            if '<img src=' in cell_val:
                # Extract image URL
                img_url_match = re.search(r'src="([^"]+)"', cell_val)
                if img_url_match:
                    img_url = img_url_match.group(1)
                    
                    # Handle relative URLs
                    if img_url.startswith('/'):
                        # Try to determine the base URL
                        if 'source_url' in table_data:
                            from urllib.parse import urlparse
                            parsed_url = urlparse(table_data['source_url'])
                            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                            img_url = base_url + img_url
                        else:
                            # Can't resolve relative URL without base
                            df.at[i, col] = f"Image URL: {img_url} (relative URL, could not resolve)"
                            processed = True
                            continue
                    
                    # Generate a unique filename for the image
                    img_extension = os.path.splitext(img_url.split('?')[0])[1] or '.jpg'
                    if img_extension.lower() not in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']:
                        img_extension = '.jpg'  # Default extension
                    
                    img_filename = f"{base_dir}/{safe_title}_{uuid.uuid4().hex[:8]}{img_extension}"
                    
                    try:
                        # Download the image
                        response = requests.get(img_url, stream=True, timeout=10)
                        if response.status_code == 200:
                            with open(img_filename, 'wb') as img_file:
                                img_file.write(response.content)
                            
                            # Update cell with local image path
                            df.at[i, col] = f"Image: {img_filename}"
                            processed = True
                        else:
                            df.at[i, col] = f"Image URL: {img_url} (download failed, status: {response.status_code})"
                            processed = True
                    except Exception as img_err:
                        print(f"Error downloading image {img_url}: {img_err}")
                        df.at[i, col] = f"Image URL: {img_url} (download failed: {str(img_err)[:100]})"
                        processed = True
    
    # Update table data with processed DataFrame if changes were made
    if processed:
        table_data['dataframe_with_local_images'] = df
    
    return table_data

def screenshot_table(url, table_title):
    """
    Legacy function for backwards compatibility - captures a screenshot of a specific table
    Args:
        url (str): The URL of the webpage containing the table
        table_title (str): The title or caption of the table to screenshot
    """
    # Set up Chrome options for Colab environment
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Initialize WebDriver with specific Colab settings
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # Load the webpage
        driver.get(url)
        print(f"Loaded page: {url}")
        
        # Wait for the page to load
        time.sleep(3)
        
        # Handle cookie consent overlays
        accept_cookies(driver)
        
        # Find the table element
        table_element = None
        
        # Strategy 1: Find by title in standard table elements
        try:
            # Look for tables with captions or titles
            tables = driver.find_elements(By.XPATH,
                f"//table[./caption[contains(text(), '{table_title}')] or @title[contains(., '{table_title}')]]")
            if tables:
                table_element = tables[0]
                print(f"Found table with caption/title containing '{table_title}'")
        except Exception as e:
            print(f"Error in strategy 1: {e}")
        
        # Strategy 2: Find heading elements containing the title, then look for nearby tables
        if not table_element:
            try:
                # Find elements containing the title text
                title_xpath = f"//*[contains(text(), '{table_title}')]"
                title_elements = driver.find_elements(By.XPATH, title_xpath)
                
                for title_el in title_elements:
                    print(f"Examining title element: {title_el.tag_name} with text: {title_el.text[:50]}...")
                    
                    # Look for tables near this title element
                    try:
                        # Try following siblings first (most common pattern)
                        following_tables = driver.find_elements(By.XPATH,
                            f"//h1[contains(text(), '{table_title}')]/following::table[1] | " +
                            f"//h2[contains(text(), '{table_title}')]/following::table[1] | " +
                            f"//h3[contains(text(), '{table_title}')]/following::table[1] | " +
                            f"//h4[contains(text(), '{table_title}')]/following::table[1] | " +
                            f"//div[contains(text(), '{table_title}')]/following::table[1]")
                        
                        if following_tables:
                            table_element = following_tables[0]
                            print(f"Found table after title element")
                            break
                        
                        # Try looking for div-based tables
                        following_div_tables = driver.find_elements(By.XPATH,
                            f"//h1[contains(text(), '{table_title}')]/following::div[contains(@class, 'table')][1] | " +
                            f"//h2[contains(text(), '{table_title}')]/following::div[contains(@class, 'table')][1] | " +
                            f"//h3[contains(text(), '{table_title}')]/following::div[contains(@class, 'table')][1] | " +
                            f"//h4[contains(text(), '{table_title}')]/following::div[contains(@class, 'table')][1] | " +
                            f"//div[contains(text(), '{table_title}')]/following::div[contains(@class, 'table')][1] | " +
                            f"//h1[contains(text(), '{table_title}')]/following::div[contains(@class, 'standings')][1] | " +
                            f"//h2[contains(text(), '{table_title}')]/following::div[contains(@class, 'standings')][1] | " +
                            f"//h3[contains(text(), '{table_title}')]/following::div[contains(@class, 'standings')][1] | " +
                            f"//h4[contains(text(), '{table_title}')]/following::div[contains(@class, 'standings')][1] | " +
                            f"//div[contains(text(), '{table_title}')]/following::div[contains(@class, 'standings')][1]")
                        
                        if following_div_tables:
                            table_element = following_div_tables[0]
                            print(f"Found div-based table after title element")
                            break
                        
                        # Try to find a table in the parent container
                        parent = title_el.find_element(By.XPATH, "..")
                        
                        # First look for standard tables
                        tables_in_parent = parent.find_elements(By.TAG_NAME, "table")
                        if tables_in_parent:
                            table_element = tables_in_parent[0]
                            print(f"Found table within title element's parent")
                            break
                        
                        # Then look for div-based tables
                        div_tables_in_parent = parent.find_elements(By.XPATH,
                            ".//div[contains(@class, 'table') or contains(@class, 'standings') or contains(@class, 'grid')]")
                        if div_tables_in_parent:
                            table_element = div_tables_in_parent[0]
                            print(f"Found div-based table within title element's parent")
                            break
                    except NoSuchElementException:
                        continue
            except Exception as e:
                print(f"Error in strategy 2: {e}")
        
        # Strategy 3: Specific handling for ESPN and similar sports sites
        if not table_element and ("espn.com" in url or "standings" in url.lower()):
            try:
                print("Using sports website specific strategy")
                # Look for tabs or filters that might match our title
                filter_xpath = "//div[contains(@class, 'filters')]//div | " + \
                              "//div[contains(@class, 'tablist')]//div | " + \
                              "//div[contains(@class, 'tabs')]//div | " + \
                              "//ul[contains(@class, 'tabs')]//li"
                filters = driver.find_elements(By.XPATH, filter_xpath)
                clicked = False
                for filter_el in filters:
                    try:
                        filter_text = filter_el.text.strip().lower()
                        if table_title.lower() in filter_text or filter_text in table_title.lower():
                            # This filter seems to match our title, try clicking it
                            filter_el.click()
                            clicked = True
                            print(f"Clicked filter/tab: {filter_text}")
                            time.sleep(2)  # Wait for content to update
                            break
                    except:
                        continue
                # Now look for standings containers
                standings_xpath = "//div[contains(@class, 'standings')] | " + \
                                 "//div[contains(@class, 'StandingsTable')] | " + \
                                 "//div[contains(@class, 'Table')] | " + \
                                 "//section[contains(@class, 'standings')]"
                standings_containers = driver.find_elements(By.XPATH, standings_xpath)
                if standings_containers:
                    # If we clicked a filter, take the first container (most likely to be relevant)
                    if clicked:
                        table_element = standings_containers[0]
                        print("Found standings container after clicking filter")
                    else:
                        # Try to find a container that might match our title
                        for container in standings_containers:
                            container_text = container.text.lower()
                            if table_title.lower() in container_text:
                                table_element = container
                                print("Found standings container matching title")
                                break
                        # If no specific match, take the first one
                        if not table_element and standings_containers:
                            table_element = standings_containers[0]
                            print("Using first standings container")
            except Exception as e:
                print(f"Error in sports website strategy: {e}")
        
        # Strategy 4: Fallback to any table-like element
        if not table_element:
            try:
                print("Using fallback strategy")
                # Try to find any HTML table
                tables = driver.find_elements(By.TAG_NAME, "table")
                if tables:
                    table_element = tables[0]
                    print(f"Fallback: Using first HTML table (of {len(tables)})")
                else:
                    # Try to find any div that looks like a table
                    table_like_divs = driver.find_elements(By.XPATH,
                        "//div[contains(@class, 'table') or " +
                        "contains(@class, 'standings') or " +
                        "contains(@class, 'grid') or " +
                        "contains(@class, 'data')]")
                    if table_like_divs:
                        table_element = table_like_divs[0]
                        print(f"Fallback: Using first div-based table (of {len(table_like_divs)})")
            except Exception as e:
                print(f"Error in fallback strategy: {e}")
        
        # If we still don't have a table element, take a screenshot of the whole page
        if not table_element:
            print("No table found. Taking screenshot of the entire page.")
            os.makedirs('screenshots', exist_ok=True)
            driver.save_screenshot("screenshots/full_page_screenshot.png")
            print("Full page screenshot saved as screenshots/full_page_screenshot.png")
            return
        
        # Scroll to the table element to make sure it's visible
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", table_element)
        time.sleep(1)
        
        # Special handling for ESPN tables
        if "espn.com" in url:
            try:
                print("Applying ESPN-specific table handling")
                # For ESPN, we may need to find the parent container that holds the full table
                # since ESPN often has separate containers for different parts of the table
                espn_table_container = table_element
                # Try to find a larger container by going up in the DOM
                for _ in range(5):  # Try up to 5 levels up
                    parent = driver.execute_script("return arguments[0].parentElement", espn_table_container)
                    if parent:
                        # Check if this parent is wider and contains the full table
                        parent_size = driver.execute_script("return {width: arguments[0].offsetWidth, height: arguments[0].offsetHeight}", parent)
                        current_size = driver.execute_script("return {width: arguments[0].offsetWidth, height: arguments[0].offsetHeight}", espn_table_container)
                        # If parent is significantly wider, use it instead
                        if parent_size['width'] > current_size['width'] * 1.3:  # 30% wider
                            espn_table_container = parent
                            print(f"Found wider parent container: {parent_size['width']}px vs {current_size['width']}px")
                        else:
                            # If we didn't find a wider container, stop looking
                            break
                    else:
                        break
                # Check if we found a better container
                if espn_table_container != table_element:
                    table_element = espn_table_container
                    print("Using wider container for ESPN table")
                # Try to find specific standings table container class names
                try:
                    full_standings = driver.find_element(By.XPATH, "//div[contains(@class, 'ResponsiveTable') or contains(@class, 'Standings')]")
                    if full_standings:
                        table_element = full_standings
                        print("Found ESPN ResponsiveTable/Standings container")
                except:
                    pass
                # Find if there's a container with all the stats (the one with column headers)
                try:
                    headers = driver.find_elements(By.XPATH, "//tr[th[contains(text(), 'GP') or contains(text(), 'W') or contains(text(), 'L') or contains(text(), 'P')]]")
                    if headers:
                        # Find the closest table or div containing this header
                        for header in headers:
                            parent = header
                            for _ in range(5):  # Look up to 5 levels up
                                parent = driver.execute_script("return arguments[0].parentElement", parent)
                                if parent and (parent.tag_name == 'table' or
                                              ('table' in parent.get_attribute('class') or
                                               'Table' in parent.get_attribute('class'))):
                                    table_element = parent
                                    print("Found table with proper headers (GP, W, L, P)")
                                    break
                except:
                    pass
            except Exception as e:
                print(f"Error in ESPN-specific handling: {e}")
        
        # Ensure the table is fully visible (if possible)
        try:
            # Adjust the window size to be large enough
            driver.set_window_size(2000, 1500)  # Use a larger window
            # Check if table is larger than viewport and adjust accordingly
            driver.execute_script("""
                var rect = arguments[0].getBoundingClientRect();
                if (rect.height > window.innerHeight) {
                    window.scrollTo(0, window.pageYOffset + rect.top - 100);
                }
                // If there are horizontal scrollbars, try to capture the full width
                if (rect.width > window.innerWidth) {
                    arguments[0].style.maxWidth = "none";
                    arguments[0].style.width = "auto";
                }
            """, table_element)
            time.sleep(1)
        except:
            pass
        
        # Create a clean filename from the table title
        safe_title = re.sub(r'[^\w\s-]', '', table_title).strip()
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        screenshot_filename = f"table_{safe_title}_screenshot.png"
        
        # For ESPN tables, try to ensure we get the full table by manipulating the page
        if "espn.com" in url:
            try:
                # Try to make table fully visible by adjusting CSS
                driver.execute_script("""
                    // Force table to be fully visible and expanded
                    arguments[0].style.overflow = 'visible';
                    arguments[0].style.maxWidth = 'none';
                    arguments[0].style.width = 'auto';
                    // If there are any parent containers with overflow:hidden, fix them
                    let parent = arguments[0].parentElement;
                    for (let i = 0; i < 10 && parent; i++) {
                        if (window.getComputedStyle(parent).overflow === 'hidden' ||
                            window.getComputedStyle(parent).overflowX === 'hidden') {
                            parent.style.overflow = 'visible';
                            parent.style.overflowX = 'visible';
                        }
                        parent = parent.parentElement;
                    }
                    // If there are any parent containers with fixed width, expand them
                    parent = arguments[0].parentElement;
                    for (let i = 0; i < 10 && parent; i++) {
                        if (window.getComputedStyle(parent).width !== 'auto') {
                            parent.style.width = 'auto';
                            parent.style.maxWidth = 'none';
                        }
                        parent = parent.parentElement;
                    }
                """, table_element)
                # Wait for changes to apply
                time.sleep(2)
            except Exception as e:
                print(f"Error adjusting table for ESPN: {e}")
        
        # Create screenshots directory if it doesn't exist
        os.makedirs('screenshots', exist_ok=True)
        
        # Take the screenshot
        try:
            # First attempt: regular screenshot
            table_element.screenshot(f"screenshots/{screenshot_filename}")
            print(f"Table screenshot saved as screenshots/{screenshot_filename}")
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            # Fallback to a different approach if the first one fails
            try:
                # Try full page screenshot instead
                driver.save_screenshot(f"screenshots/full_{screenshot_filename}")
                print(f"Full page screenshot saved as screenshots/full_{screenshot_filename}")
            except:
                pass
    
    except Exception as e:
        print(f"An error occurred: {e}")
        # Take screenshot of the entire page as a fallback
        try:
            os.makedirs('screenshots', exist_ok=True)
            driver.save_screenshot("screenshots/error_screenshot.png")
            print("Error occurred, full page screenshot saved as screenshots/error_screenshot.png")
        except:
            pass
    
    finally:
        driver.quit()

# For testing purposes
if __name__ == "__main__":
    # Example usage
    url = "https://www.worldometers.info/world-population/"
    print(f"Extracting tables from {url}")
    tables = screenshot_tables(url)
    print(f"Extracted {len(tables)} tables")
    
    # Print table information
    for i, table in enumerate(tables):
        print(f"\nTable {i+1}: {table['title']}")
        if 'dataframe' in table:
            print(f"Columns: {list(table['dataframe'].columns)}")
            print(f"Rows: {len(table['dataframe'])}")
        if 'screenshot' in table:
            print(f"Screenshot: {table['screenshot']}")
        
        # Process any images in the table
        if table.get('has_images', False):
            print("Table contains images - downloading...")
            processed_table = download_table_images(table)
            if 'dataframe_with_local_images' in processed_table:
                print("Successfully processed images in table")