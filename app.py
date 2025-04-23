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
import uuid
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, urljoin
from collections import Counter

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
        # Common IDs and classes
        "//*[contains(@id, 'cookie-accept') or contains(@id, 'accept-cookie') or contains(@id, 'cookieAccept')]",
        "//*[contains(@class, 'cookie-accept') or contains(@class, 'accept-cookie') or contains(@class, 'cookieAccept')]"
    ]
    
    # Try different patterns to find and click consent buttons
    for attempt in range(max_attempts):
        for xpath in consent_button_patterns:
            try:
                buttons = driver.find_elements(By.XPATH, xpath)
                for button in buttons:
                    if button.is_displayed():
                        button.click()
                        print("Clicked consent button")
                        time.sleep(1)
                        return True
            except Exception:
                # Continue to next pattern if error occurs
                pass
        
        # Wait before trying again
        if attempt < max_attempts - 1:
            time.sleep(1)
    
    # Check for cookie banners in iframes
    try:
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        for iframe in iframes:
            try:
                # Check if iframe might contain cookie consent
                iframe_id = iframe.get_attribute("id") or ""
                iframe_name = iframe.get_attribute("name") or ""
                
                if any(term in iframe_id.lower() or term in iframe_name.lower() 
                       for term in ["cookie", "consent", "privacy"]):
                    driver.switch_to.frame(iframe)
                    
                    # Try to find consent buttons in the iframe
                    for xpath in consent_button_patterns:
                        try:
                            buttons = driver.find_elements(By.XPATH, xpath)
                            for button in buttons:
                                if button.is_displayed():
                                    button.click()
                                    driver.switch_to.default_content()
                                    return True
                        except:
                            pass
                    
                    driver.switch_to.default_content()
            except:
                driver.switch_to.default_content()
    except:
        pass
    
    print("No cookie consent buttons found or handled")
    return False

def capture_element_screenshot(driver, element, filename):
    """
    Capture a screenshot of a specific element with improved handling for full visibility
    Args:
        driver: Selenium WebDriver instance
        element: WebElement to screenshot
        filename: File path to save the screenshot
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure the element is in the viewport
        driver.execute_script("arguments[0].scrollIntoView({block: 'center', behavior: 'instant'});", element)
        time.sleep(0.5)
        
        # Try to ensure the whole element is visible
        driver.execute_script("""
            // Make element fully visible
            arguments[0].style.overflow = 'visible';
            
            // Fix parent containers with overflow issues
            let parent = arguments[0].parentElement;
            for (let i = 0; i < 5 && parent; i++) {
                if (window.getComputedStyle(parent).overflow === 'hidden' ||
                    window.getComputedStyle(parent).overflowX === 'hidden') {
                    parent.style.overflow = 'visible';
                    parent.style.overflowX = 'visible';
                }
                parent = parent.parentElement;
            }
            
            // Handle fixed-width containers
            arguments[0].style.maxWidth = 'none';
            arguments[0].style.width = 'auto';
        """, element)
        
        time.sleep(0.5)
        
        # Take the screenshot
        element.screenshot(filename)
        print(f"Screenshot saved: {filename}")
        return True
        
    except Exception as e:
        print(f"Error capturing element screenshot: {e}")
        return False

def find_table_containers(driver, url):
    """
    Find all elements that potentially contain tables on a webpage
    Args:
        driver: Selenium WebDriver instance
        url: URL of the webpage
    Returns:
        list: List of potential table container elements
    """
    potential_containers = []
    
    # 1. Find actual HTML tables
    print("Looking for HTML tables...")
    tables = driver.find_elements(By.TAG_NAME, "table")
    for table in tables:
        if table.is_displayed():
            # Only consider tables with actual content
            try:
                rows = table.find_elements(By.TAG_NAME, "tr")
                if len(rows) > 1:  # Need at least a header and data row
                    potential_containers.append({
                        'element': table,
                        'type': 'html_table',
                        'tag': 'table'
                    })
            except:
                pass
    
    # 2. Find div-based tables by common class names
    print("Looking for div-based tables...")
    table_classes = [
        ".table", ".datatable", ".data-table", ".grid-table", 
        ".table-responsive", ".table-container", ".tableWrapper",
        "[class*='table']", "[class*='Table']", "[class*='grid']",
        "[class*='Grid']", "[role='table']", "[role='grid']"
    ]
    
    for selector in table_classes:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                # Skip if this element is a child of another element we already found
                if element.is_displayed() and not any(
                    is_child_element(driver, element, container['element']) 
                    for container in potential_containers
                ):
                    potential_containers.append({
                        'element': element,
                        'type': 'div_table',
                        'tag': element.tag_name,
                        'class': element.get_attribute('class')
                    })
        except:
            pass
    
    # 3. Find containers with grid structure (often used for tables)
    grid_selectors = [
        "div.grid", "div.cards", "div.list-view", "ul.grid", 
        "[class*='card-container']", "[class*='item-grid']",
        "[style*='display: grid']", "[style*='display:grid']"
    ]
    
    for selector in grid_selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                if element.is_displayed() and not any(
                    is_child_element(driver, element, container['element']) 
                    for container in potential_containers
                ):
                    potential_containers.append({
                        'element': element,
                        'type': 'grid',
                        'tag': element.tag_name,
                        'class': element.get_attribute('class')
                    })
        except:
            pass
    
    # 4. Find any elements with multiple child elements that look like rows
    print("Looking for row-based structures...")
    container_selectors = ["div", "section", "article", "main", "aside"]
    
    for container_tag in container_selectors:
        try:
            containers = driver.find_elements(By.TAG_NAME, container_tag)
            for container in containers:
                if not container.is_displayed():
                    continue
                
                # Skip if this element is a child of another element we already found
                if any(is_child_element(driver, container, existing['element']) 
                       for existing in potential_containers):
                    continue
                
                # Check if container has multiple similar children that could be rows
                row_items = []
                
                # Try different potential row patterns
                for row_selector in [".//div[contains(@class, 'row')]", ".//li", ".//tr", ".//ul/li"]:
                    try:
                        items = container.find_elements(By.XPATH, row_selector)
                        if len(items) > 3:  # Need multiple rows to be a table
                            row_items = items
                            break
                    except:
                        pass
                
                if row_items:
                    potential_containers.append({
                        'element': container,
                        'type': 'row_container',
                        'tag': container.tag_name,
                        'class': container.get_attribute('class'),
                        'row_count': len(row_items)
                    })
        except:
            pass
    
    print(f"Found {len(potential_containers)} potential table containers")
    return potential_containers

def is_child_element(driver, element, potential_parent):
    """
    Check if an element is a child of another element
    Args:
        driver: Selenium WebDriver instance
        element: Element to check
        potential_parent: Potential parent element
    Returns:
        bool: True if element is a child of potential_parent
    """
    try:
        # Use JavaScript to check parent-child relationship
        return driver.execute_script("""
            function isDescendant(child, parent) {
                let node = child;
                while (node != null) {
                    if (node == parent) {
                        return true;
                    }
                    node = node.parentNode;
                }
                return false;
            }
            return isDescendant(arguments[0], arguments[1]);
        """, element, potential_parent)
    except:
        return False

def extract_table_data(driver, container):
    """
    Extract structured data from a table container
    Args:
        driver: Selenium WebDriver instance
        container: Dictionary with information about the table container
    Returns:
        dict: Table data including DataFrame, or None if extraction failed
    """
    try:
        element = container['element']
        
        # Get container's HTML
        html = element.get_attribute('outerHTML')
        
        # Find table title from nearby headings
        title = get_table_title(driver, element)
        
        # Use BeautifulSoup to parse the HTML structure
        soup = BeautifulSoup(html, 'html.parser')
        
        # Different extraction strategies based on container type
        if container['type'] == 'html_table':
            # Standard HTML table extraction
            headers, rows = extract_html_table_data(soup)
        elif container['type'] == 'div_table' or container['type'] == 'grid':
            # Div-based table or grid extraction
            headers, rows = extract_div_table_data(soup)
        else:
            # Row-based container extraction
            headers, rows = extract_row_container_data(soup)
        
        # Create DataFrame from extracted data
        df = None
        if rows:
            if headers and len(headers) == len(rows[0]):
                df = pd.DataFrame(rows, columns=headers)
            else:
                # Create generic headers if needed
                max_cols = max(len(row) for row in rows)
                headers = [f"Column {i+1}" for i in range(max_cols)]
                # Pad rows with empty strings if needed
                padded_rows = [row + [''] * (max_cols - len(row)) for row in rows]
                df = pd.DataFrame(padded_rows, columns=headers)
        
        # Only return if we have a valid DataFrame with data
        if df is not None and not df.empty:
            # Check if table contains images
            has_images = check_for_images(soup, df)
            
            return {
                'title': title,
                'dataframe': df,
                'html': html,
                'has_images': has_images,
                'container_type': container['type']
            }
        
        return None
    
    except Exception as e:
        print(f"Error extracting table data: {e}")
        return None

def get_table_title(driver, element):
    """
    Find the title for a table element by looking for nearby headings
    Args:
        driver: Selenium WebDriver instance
        element: Table container element
    Returns:
        str: Table title or generic title if none found
    """
    # Try to find a caption or title directly
    try:
        caption = element.find_element(By.TAG_NAME, "caption")
        if caption:
            return caption.text.strip()
    except:
        pass
    
    # Look for nearby headings using JavaScript
    try:
        title = driver.execute_script("""
            function findTableHeading(element) {
                // Check for heading elements before the table
                let prevEl = element.previousElementSibling;
                while (prevEl) {
                    if (prevEl.tagName.match(/^H[1-6]$/)) {
                        return prevEl.textContent.trim();
                    }
                    prevEl = prevEl.previousElementSibling;
                }
                
                // Check parent container for headings before this element
                let parent = element.parentElement;
                if (parent) {
                    for (let i = 0; i < parent.children.length; i++) {
                        let child = parent.children[i];
                        if (child === element) {
                            break;
                        }
                        if (child.tagName && child.tagName.match(/^H[1-6]$/)) {
                            return child.textContent.trim();
                        }
                    }
                }
                
                // Check for closest heading anywhere above
                const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
                let closestHeading = null;
                let closestDistance = Infinity;
                
                const rect = element.getBoundingClientRect();
                const tableTop = rect.top + window.scrollY;
                
                for (let heading of headings) {
                    const headingRect = heading.getBoundingClientRect();
                    const headingTop = headingRect.top + window.scrollY;
                    
                    if (headingTop < tableTop) {
                        const distance = tableTop - headingTop;
                        if (distance < closestDistance) {
                            closestDistance = distance;
                            closestHeading = heading;
                        }
                    }
                }
                
                if (closestHeading && closestDistance < 500) {
                    return closestHeading.textContent.trim();
                }
                
                return "";
            }
            return findTableHeading(arguments[0]);
        """, element)
        
        if title:
            return title
    except:
        pass
    
    # If no title found, use element attributes or generic name
    try:
        element_id = element.get_attribute('id')
        if element_id:
            return f"Table: {element_id}"
        
        element_class = element.get_attribute('class')
        if element_class:
            clean_class = element_class.split()[0]
            return f"Table: {clean_class}"
    except:
        pass
    
    # Generate a unique title as fallback
    return f"Table {uuid.uuid4().hex[:6]}"

def extract_html_table_data(soup):
    """
    Extract data from standard HTML table
    Args:
        soup: BeautifulSoup object of the table HTML
    Returns:
        tuple: (headers, rows) where headers is a list of column names and rows is a list of data rows
    """
    headers = []
    rows = []
    
    # Find the table element
    table = soup.find('table')
    if not table:
        return headers, rows
    
    # Extract headers from thead
    thead = table.find('thead')
    if thead:
        th_row = thead.find('tr')
        if th_row:
            headers = [cell.get_text(strip=True) for cell in th_row.find_all(['th', 'td'])]
    
    # If no headers in thead, try first row
    if not headers:
        first_row = table.find('tr')
        if first_row:
            # Check if first row looks like a header
            cells = first_row.find_all(['th', 'td'])
            if any(cell.name == 'th' for cell in cells):
                headers = [cell.get_text(strip=True) for cell in cells]
    
    # Extract data rows from tbody or table
    tbody = table.find('tbody') or table
    
    # Process all rows
    for tr in tbody.find_all('tr'):
        # Skip if this is the header row we already processed
        if headers and tr == table.find('tr'):
            continue
            
        row = []
        for cell in tr.find_all(['td', 'th']):
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
    
    return headers, rows

def extract_div_table_data(soup):
    """
    Extract data from a div-based table or grid
    Args:
        soup: BeautifulSoup object of the container HTML
    Returns:
        tuple: (headers, rows) where headers is a list of column names and rows is a list of data rows
    """
    headers = []
    rows = []
    
    # Look for headers in various ways
    header_elements = soup.select('.header, .heading, .headers, th, [role="columnheader"]')
    if header_elements:
        headers = [h.get_text(strip=True) for h in header_elements]
    
    # Look for rows
    row_elements = soup.select('.row, .data-row, tr, li, [role="row"]')
    
    for row_el in row_elements:
        # Skip if this looks like a header row
        if ('header' in row_el.get('class', []) or 
            row_el.select_one('.header, .heading, th')):
            if not headers:
                headers = [cell.get_text(strip=True) for cell in row_el.select('.cell, .column, td, th, span')]
            continue
            
        # Get cell data
        cell_elements = row_el.select('.cell, .column, td, th, div, span')
        
        # If no specific cell markers, try direct children
        if not cell_elements:
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
    
    # If no rows yet, try looking for consistent structure
    if not rows:
        # Find main container
        container = soup.find()
        
        # Look for sets of similar children
        direct_children = [child for child in container.children if child.name]
        
        # Group by tag name
        child_tags = {}
        for child in direct_children:
            if child.name:
                if child.name not in child_tags:
                    child_tags[child.name] = []
                child_tags[child.name].append(child)
        
        # Find the tag with most occurrences
        most_common_tag = max(child_tags.items(), key=lambda x: len(x[1]), default=(None, []))
        
        if most_common_tag[0] and len(most_common_tag[1]) > 1:
            # First element might be header
            row_items = most_common_tag[1]
            
            # Extract headers from first item if we don't have them yet
            if not headers:
                first_item = row_items[0]
                headers = []
                for cell in first_item.children:
                    if cell.name:
                        headers.append(cell.get_text(strip=True))
                
                # Process remaining items as rows
                for item in row_items[1:]:
                    row = []
                    for cell in item.children:
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
                    
                    # Only add non-empty rows
                    if row and any(str(cell).strip() for cell in row):
                        rows.append(row)
    
    return headers, rows

def extract_row_container_data(soup):
    """
    Extract data from a container with row-like elements
    Args:
        soup: BeautifulSoup object of the container HTML
    Returns:
        tuple: (headers, rows) where headers is a list of column names and rows is a list of data rows
    """
    headers = []
    rows = []
    
    # Try to find rows
    row_selectors = [
        'li', 'tr', 'div.row', '[role="row"]', 
        '[class*="row"]', '[class*="item"]'
    ]
    
    row_elements = []
    for selector in row_selectors:
        elements = soup.select(selector)
        if len(elements) > 2:  # Need multiple rows
            row_elements = elements
            break
    
    if not row_elements:
        return headers, rows
    
    # First element might be header
    first_row = row_elements[0]
    
    # Check if it looks like a header
    if (first_row.find('th') or 
        'header' in first_row.get('class', []) or
        first_row.select_one('[class*="header"], [class*="heading"]')):
        
        # Extract header cells
        cell_elements = first_row.find_all(['th', 'td', 'div', 'span'])
        if not cell_elements:
            cell_elements = [child for child in first_row.children if child.name]
            
        headers = [cell.get_text(strip=True) for cell in cell_elements]
        
        # Process remaining rows
        for row_el in row_elements[1:]:
            row = []
            
            # Find cells in this row
            cell_elements = row_el.find_all(['td', 'div', 'span'])
            if not cell_elements:
                cell_elements = [child for child in row_el.children if child.name]
                
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
    else:
        # No clear header, treat all rows as data
        for row_el in row_elements:
            row = []
            
            # Find cells in this row
            cell_elements = row_el.find_all(['td', 'div', 'span'])
            if not cell_elements:
                cell_elements = [child for child in row_el.children if child.name]
                
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
    
    return headers, rows

def check_for_images(soup, df):
    """
    Check if the table contains images
    Args:
        soup: BeautifulSoup object
        df: DataFrame of the table data
    Returns:
        bool: True if images are found
    """
    # Check if the HTML contains any img tags
    images = soup.find_all('img')
    if images:
        return True
    
    # Check if any cell in the DataFrame contains an image tag
    for column in df.columns:
        for value in df[column]:
            if isinstance(value, str) and '<img src=' in value:
                return True
    
    return False

def process_table_images(table_data, url, base_dir="extracted_images"):
    """
    Process and download all images in a table
    Args:
        table_data: Table data dictionary
        url: Source URL of the webpage
        base_dir: Directory to save downloaded images
    Returns:
        Updated table data with local image paths
    """
    if not table_data.get('has_images', False) or 'dataframe' not in table_data:
        return table_data
    
    # Create image directory if needed
    os.makedirs(base_dir, exist_ok=True)
    
    # Get base URL for resolving relative paths
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    # Create safe filename prefix
    safe_title = re.sub(r'[^\w\s-]', '', table_data['title']).strip().replace(' ', '_')
    safe_title = re.sub(r'[-_]+', '_', safe_title)
    
    # Create copy of the DataFrame
    df = table_data['dataframe'].copy()
    processed = False
    
    # Process each cell
    for i, row in df.iterrows():
        for col in df.columns:
            cell_val = str(row[col])
            if '<img src=' in cell_val:
                # Extract image URL
                img_url_match = re.search(r'src="([^"]+)"', cell_val)
                if img_url_match:
                    img_url = img_url_match.group(1)
                    
                    # Handle relative URLs
                    if img_url.startswith('/') or not img_url.startswith(('http://', 'https://')):
                        img_url = urljoin(base_url, img_url)
                    
                    # Generate unique image filename
                    img_extension = os.path.splitext(img_url.split('?')[0])[1] or '.jpg'
                    if img_extension.lower() not in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']:
                        img_extension = '.jpg'  # Default extension
                    
                    img_filename = f"{base_dir}/{safe_title}_{uuid.uuid4().hex[:8]}{img_extension}"
                    
                    # Download the image
                    try:
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

def screenshot_tables(url):
    """
    Capture all tables from a webpage
    Args:
        url (str): URL of the webpage to analyze
    Returns:
        list: List of dictionaries containing table data
    """
    # Set up Chrome options for headless browser
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Initialize WebDriver
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_window_size(1920, 1080)
    
    # Track results
    tables_info = []
    
    try:
        # Load the webpage
        print(f"Loading webpage: {url}")
        driver.get(url)
        
        # Wait for page to load and handle any cookie consent popups
        time.sleep(5)
        accept_cookies(driver)
        
        # Create a safe domain name for filenames
        domain_name = re.sub(r'^https?://(www\.)?', '', url).split('/')[0]
        safe_domain = re.sub(r'[^\w\s-]', '', domain_name).strip().replace('.', '_')
        
        # Ensure screenshots directory exists
        os.makedirs('screenshots', exist_ok=True)
        
        # Take full page screenshot for reference
        full_page_file = f"screenshots/{safe_domain}_full_page.png"
        driver.save_screenshot(full_page_file)
        print(f"Full page screenshot saved: {full_page_file}")
        
        # Find all potential table containers
        containers = find_table_containers(driver, url)
        
        # Process each potential table container
        for idx, container in enumerate(containers):
            try:
                # Get element and check if it's still valid and visible
                element = container['element']
                if not element.is_displayed():
                    continue
                
                # Take a screenshot of this container
                screenshot_file = f"screenshots/{safe_domain}_table_{idx+1}.png"
                if capture_element_screenshot(driver, element, screenshot_file):
                    # Extract data from the container
                    table_data = extract_table_data(driver, container)
                    
                    if table_data:
                        # Add source URL and screenshot info
                        table_data['source_url'] = url
                        table_data['screenshot'] = screenshot_file
                        
                        # Check for real data (more than just headers)
                        if len(table_data['dataframe']) > 0:
                            # Process images if present
                            if table_data.get('has_images', False):
                                table_data = process_table_images(table_data, url)
                            
                            tables_info.append(table_data)
                            print(f"Successfully extracted table {idx+1}: {table_data['title']}")
            
            except Exception as e:
                print(f"Error processing container {idx+1}: {e}")
                continue
        
        print(f"Total tables extracted: {len(tables_info)}")
        
        # If no tables were extracted, use AI on the full page screenshot later
        if not tables_info:
            print("No tables were successfully extracted. Full page screenshot can be used for AI extraction.")
            tables_info.append({
                'title': f"Full page - {domain_name}",
                'screenshot': full_page_file,
                'source_url': url,
                'dataframe': None,
                'full_page': True
            })
        
        return tables_info
    
    except Exception as e:
        print(f"Error during table extraction: {e}")
        import traceback
        traceback.print_exc()
        
        # Create error screenshot
        try:
            os.makedirs('screenshots', exist_ok=True)
            error_file = f"screenshots/error_{safe_domain}.png"
            driver.save_screenshot(error_file)
            print(f"Error screenshot saved: {error_file}")
            
            # Return error info
            return [{
                'title': f"Error - {domain_name}",
                'screenshot': error_file,
                'source_url': url,
                'dataframe': None,
                'error': str(e),
                'full_page': True
            }]
        except:
            return []
    
    finally:
        # Always close the browser
        driver.quit()

def screenshot_table(url, table_title):
    """
    Legacy function for backwards compatibility - captures a screenshot of a specific table
    Args:
        url (str): The URL of the webpage containing the table
        table_title (str): The title or caption of the table to screenshot
    """
    # Set up Chrome options for headless browser
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Initialize WebDriver
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_window_size(1920, 1080)
    
    try:
        # Load the webpage
        driver.get(url)
        print(f"Loaded page: {url}")
        
        # Wait for the page to load and handle cookie popups
        time.sleep(5)
        accept_cookies(driver)
        
        # Find potential table by title
        found_element = None
        
        # Strategy 1: Find headings matching the title
        try:
            # Look for headings containing the title
            heading_xpath = f"//h1[contains(text(), '{table_title}')] | //h2[contains(text(), '{table_title}')] | //h3[contains(text(), '{table_title}')] | //h4[contains(text(), '{table_title}')]"
            heading_elements = driver.find_elements(By.XPATH, heading_xpath)
            
            for heading in heading_elements:
                # Look for tables after this heading
                script = """
                    function findTableAfterHeading(heading) {
                        // Check siblings after the heading
                        let element = heading;
                        while (element = element.nextElementSibling) {
                            // Check if this element is a table
                            if (element.tagName === 'TABLE') {
                                return element;
                            }
                            
                            // Check if this element contains a table
                            const tables = element.querySelectorAll('table');
                            if (tables.length > 0) {
                                return tables[0];
                            }
                            
                            // Check for div-based tables
                            const divTables = element.querySelectorAll('.table, [class*="table"], [role="table"], [class*="grid"]');
                            if (divTables.length > 0) {
                                return divTables[0];
                            }
                        }
                        
                        // Check parent's siblings
                        let parent = heading.parentElement;
                        if (parent) {
                            let parentIndex = Array.from(parent.parentElement.children).indexOf(parent);
                            let siblings = Array.from(parent.parentElement.children).slice(parentIndex + 1);
                            
                            for (let sibling of siblings) {
                                // Check if this sibling contains a table
                                const tables = sibling.querySelectorAll('table, .table, [class*="table"], [role="table"], [class*="grid"]');
                                if (tables.length > 0) {
                                    return tables[0];
                                }
                            }
                        }
                        
                        return null;
                    }
                    return findTableAfterHeading(arguments[0]);
                """
                
                potential_table = driver.execute_script(script, heading)
                if potential_table:
                    found_element = potential_table
                    print(f"Found table after heading: {table_title}")
                    break
        except Exception as e:
            print(f"Error in heading search strategy: {e}")
        
        # Strategy 2: Search for tables with captions matching the title
        if not found_element:
            try:
                caption_xpath = f"//table[./caption[contains(text(), '{table_title}')]]"
                caption_tables = driver.find_elements(By.XPATH, caption_xpath)
                if caption_tables:
                    found_element = caption_tables[0]
                    print(f"Found table with caption: {table_title}")
            except Exception as e:
                print(f"Error in caption search strategy: {e}")
        
        # Strategy 3: Search for table containers with the title text
        if not found_element:
            try:
                # Find containers with class containing 'table' and the title text
                container_xpath = f"//*[contains(@class, 'table') and contains(text(), '{table_title}')]"
                containers = driver.find_elements(By.XPATH, container_xpath)
                if containers:
                    found_element = containers[0]
                    print(f"Found table container with title text: {table_title}")
            except Exception as e:
                print(f"Error in container search strategy: {e}")
        
        # Fallback: Use first table on the page
        if not found_element:
            try:
                tables = driver.find_elements(By.TAG_NAME, "table")
                if tables:
                    found_element = tables[0]
                    print("Fallback: Using first table on page")
                else:
                    # Try div-based tables
                    div_tables = driver.find_elements(By.CSS_SELECTOR, ".table, [class*='table'], [role='table']")
                    if div_tables:
                        found_element = div_tables[0]
                        print("Fallback: Using first div-based table")
            except Exception as e:
                print(f"Error in fallback strategy: {e}")
        
        # If no table found, take screenshot of the whole page
        if not found_element:
            print("No matching table found. Taking screenshot of the entire page.")
            os.makedirs('screenshots', exist_ok=True)
            driver.save_screenshot("screenshots/full_page_screenshot.png")
            print("Full page screenshot saved as screenshots/full_page_screenshot.png")
            return
        
        # Create a clean filename from the table title
        safe_title = re.sub(r'[^\w\s-]', '', table_title).strip()
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        screenshot_filename = f"table_{safe_title}_screenshot.png"
        
        # Ensure screenshots directory exists
        os.makedirs('screenshots', exist_ok=True)
        
        # Take the screenshot
        capture_element_screenshot(driver, found_element, f"screenshots/{screenshot_filename}")
        print(f"Table screenshot saved as screenshots/{screenshot_filename}")
    
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
