import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from collections import deque


class SimpleCrawler:
    def __init__(self, delay=1, max_pages=10):
        self.delay = delay  # Delay between requests (be respectful)
        self.max_pages = max_pages
        self.visited = set()
        self.to_visit = deque()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def is_valid_url(self, url):
        """Check if URL is valid and should be crawled"""
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc) and bool(parsed.scheme)
        except:
            return False

    def get_links(self, url, html):
        """Extract all links from HTML content"""
        links = set()
        try:
            soup = BeautifulSoup(html, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                if self.is_valid_url(full_url):
                    links.add(full_url)
        except Exception as e:
            print(f"Error extracting links from {url}: {e}")
        return links

    def fetch_page(self, url):
        """Fetch a single page and return content"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def crawl(self, start_url, same_domain=True):
        """Main crawling function"""
        if not self.is_valid_url(start_url):
            print(f"Invalid starting URL: {start_url}")
            return

        start_domain = urlparse(start_url).netloc
        self.to_visit.append(start_url)

        print(f"Starting crawl from: {start_url}")
        print(f"Max pages: {self.max_pages}")
        print(f"Same domain only: {same_domain}")
        print("-" * 50)

        while self.to_visit and len(self.visited) < self.max_pages:
            current_url = self.to_visit.popleft()

            if current_url in self.visited:
                continue

            # Check domain restriction
            if same_domain and urlparse(current_url).netloc != start_domain:
                continue

            print(f"Crawling: {current_url}")

            # Fetch the page
            html = self.fetch_page(current_url)
            if html is None:
                continue

            self.visited.add(current_url)

            # Process the page (you can customize this)
            self.process_page(current_url, html)

            # Extract links and add to queue
            links = self.get_links(current_url, html)
            for link in links:
                if link not in self.visited:
                    self.to_visit.append(link)

            # Be respectful - add delay
            time.sleep(self.delay)

        print(f"\nCrawl completed. Visited {len(self.visited)} pages.")

    def process_page(self, url, html):
        """Process each page - customize this method"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title"

            # Count links
            links = soup.find_all('a', href=True)

            print(f"  Title: {title_text[:60]}...")
            print(f"  Links found: {len(links)}")
            print(f"  Page size: {len(html)} bytes")

            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)

            # You can add more processing here:
            # - Save content to file
            # - Extract specific data
            # - Store in database
            # - etc.

        except Exception as e:
            print(f"  Error processing page: {e}")


# Example usage
if __name__ == "__main__":
    # Create crawler instance
    crawler = SimpleCrawler(delay=1, max_pages=5)

    # Start crawling (replace with your target URL)
    start_url = "https://example.com"

    # Crawl only within the same domain
    crawler.crawl(start_url, same_domain=True)

    # Print summary
    print(f"\nSummary:")
    print(f"Total pages visited: {len(crawler.visited)}")
    print("Visited URLs:")
    for url in list(crawler.visited)[:10]:  # Show first 10
        print(f"  - {url}")
    if len(crawler.visited) > 10:
        print(f"  ... and {len(crawler.visited) - 10} more")