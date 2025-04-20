"""
Run this CMD in terminal before you run the code: pip install selenium webdriver-manager
"""
"""
uses Selenium to capture screenshots of specific elements (tables, images, graphs)
from a webpage and saves them in organized directories along with the full page HTML.
"""

# Import core Python modules
import os
import time
import logging
import shutil
import hashlib
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

# Ask for website URL at runtime
website_url = input("Enter the website URL: ")

# Configuration
CONFIG = {
    'url': website_url,
    'output_dir': "webpage_elements",
    'fullpage_dir': "full_page",
    'html_save_path': "saved_page.html",
    'min_container_width': 300,  # Minimum width in pixels
    'min_container_height': 150,  # Minimum height in pixels
    'container_selectors': {
        'image_containers': [
            '//figure[contains(@class, "mw-default-size")]',
            '//div[contains(@class, "thumb")]',
            '//div[contains(@class, "image-container")]'
        ],
        'tables': '//table[not(ancestor::div[contains(@class, "navbox")])]',
        'graphs': '//div[contains(@class, "chart") or contains(@class, "graph")]'
    },
    'exclude_selectors': [
        '.navbox', 
        '.sidebar', 
        '.mw-logo',
        '.vector-header-container'
    ],
    'scroll_padding': 50,
    'wait_timeout': 15
}

captured_hashes = set()

def initialize_environment():
    """Set up directory structure and clean previous runs"""
    try:
        if os.path.exists(CONFIG['output_dir']):
            shutil.rmtree(CONFIG['output_dir'])
            
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(CONFIG['output_dir'], CONFIG['fullpage_dir']), exist_ok=True)
        
        logging.info("Environment initialized")
        
    except Exception as e:
        logging.error(f"Environment setup failed: {str(e)}")
        raise

def create_driver():
    """Configure and initialize headless Chrome"""
    try:
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(30)
        
        logging.info("Chrome instance created")
        return driver
    
    except Exception as e:
        logging.error(f"Driver creation failed: {str(e)}")
        raise

def get_container_hash(element):
    """Create unique identifier for containers to prevent duplicates"""
    location = element.location
    size = element.size
    hash_string = f"{location['x']}_{location['y']}_{size['width']}_{size['height']}"
    return hashlib.md5(hash_string.encode()).hexdigest()

def is_valid_container(element):
    """Validate container meets size requirements and visibility"""
    try:
        size = element.size
        return all([
            element.is_displayed(),
            size['width'] >= CONFIG['min_container_width'],
            size['height'] >= CONFIG['min_container_height']
        ])
    except Exception:
        return False

def save_page_html(driver):
    """Save the HTML source code of the page"""
    try:
        with open(CONFIG['html_save_path'], 'w', encoding='utf-8') as file:
            file.write(driver.page_source)
        logging.info(f"Page HTML saved: {CONFIG['html_save_path']}")
    except Exception as e:
        logging.error(f"Saving HTML failed: {str(e)}")

def capture_full_page_screenshot(driver):
    """Capture full page screenshot by scrolling and stitching images"""
    try:
        total_width = driver.execute_script("return document.body.scrollWidth")
        total_height = driver.execute_script("return document.body.scrollHeight")
        viewport_height = driver.execute_script("return window.innerHeight")
        driver.set_window_size(total_width, viewport_height)

        timestamp = int(time.time())
        fullpage_path = os.path.join(CONFIG['output_dir'], CONFIG['fullpage_dir'], f"fullpage_{timestamp}.png")

        stitched_image = Image.new('RGB', (total_width, total_height))
        current_y = 0

        for y in range(0, total_height, viewport_height):
            driver.execute_script(f"window.scrollTo(0, {y})")
            time.sleep(0.5)
            temp_path = f"temp_screenshot_{y}.png"
            driver.save_screenshot(temp_path)
            screenshot = Image.open(temp_path)
            stitched_image.paste(screenshot, (0, current_y))
            current_y += screenshot.size[1]
            os.remove(temp_path)

        stitched_image.save(fullpage_path)
        logging.info(f"Full page screenshot saved: {fullpage_path}")
    except Exception as e:
        logging.error(f"Full page capture failed: {str(e)}")

def capture_container(driver, container, container_type):
    """Capture screenshot of validated container"""
    try:
        container_hash = get_container_hash(container)
        if container_hash in captured_hashes or not is_valid_container(container):
            return
            
        # Scroll to container
        driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", container)
        time.sleep(0.5)  # Allow for rendering
        
        # Create output filename
        timestamp = int(time.time())
        filename = f"{container_type}_{timestamp}_{container_hash[:6]}.png"
        output_path = os.path.join(CONFIG['output_dir'], filename)
        
        # Capture and save
        container.screenshot(output_path)
        captured_hashes.add(container_hash)
        logging.info(f"Captured {container_type} container: {filename}")
        
    except Exception as e:
        logging.warning(f"Container capture failed: {str(e)}")

def find_content_containers(driver):
    """Locate relevant content containers excluding navigation elements"""
    logging.info("Identifying content containers")
    
    containers = []
    
    # Find image containers
    for xpath in CONFIG['container_selectors']['image_containers']:
        containers.extend(driver.find_elements(By.XPATH, xpath))
    
    # Find tables and graphs
    for element_type in ['tables', 'graphs']:
        containers.extend(driver.find_elements(By.XPATH, CONFIG['container_selectors'][element_type]))
    
    # Filter out excluded elements
    filtered = []
    for container in containers:
        try:
            if not container.find_elements(By.XPATH, 'ancestor::*[{}]'.format(
                ' or '.join([f'contains(@class, "{cls}")' for cls in CONFIG['exclude_selectors']])
            )):
                filtered.append(container)
        except Exception:
            continue
            
    logging.info(f"Found {len(filtered)} valid containers")
    return filtered

def process_page(driver):
    """Main processing workflow"""
    try:

        # Save HTML content
        save_page_html(driver)

        # Capture full page first
        capture_full_page_screenshot(driver)
        
        # Find and process containers
        containers = find_content_containers(driver)
        for idx, container in enumerate(containers, 1):
            container_type = "image" if idx <= len(CONFIG['container_selectors']['image_containers']) else "content"
            capture_container(driver, container, container_type)
            
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise

def execute_workflow():
    """Main execution controller"""
    driver = None
    try:
        initialize_environment()
        driver = create_driver()
        
        logging.info(f"Loading: {CONFIG['url']}")
        driver.get(CONFIG['url'])
        
        WebDriverWait(driver, CONFIG['wait_timeout']).until(
            EC.presence_of_element_located((By.XPATH, "//body"))
        )
        process_page(driver)
        logging.info("Operation completed successfully")
        
    except Exception as e:
        logging.error(f"Workflow failed: {str(e)}")
    finally:
        if driver:
            driver.quit()
            logging.info("Browser instance closed")

if __name__ == "__main__":
    execute_workflow()
