import requests
from bs4 import BeautifulSoup
import polars as pl
import uuid
from urllib.parse import urljoin


class GeneralWebCrawler:
    def __init__(self):
        self.visited_urls = set()
        self.data = pl.DataFrame(
            {
                "UUID": [],
                "URL": [],
                "TEXT": [],
            }
        )

    def scrape_url(self, url):
        # Check if the URL has already been visited
        if url in self.visited_urls:
            return

        # Add the URL to visited URLs
        self.visited_urls.add(url)

        # Make a request to the URL
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to scrape URL: {url}")
            return

        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract text from HTML elements
        text = self.extract_text(soup)

        # Generate a UUID for the URL
        uid = str(uuid.uuid4())

        # Store the data in the DataFrame
        self.data = self.data.append(
            pl.DataFrame(
                {
                    "UUID": [uid] * len(text),
                    "URL": [url] * len(text),
                    "TEXT": text,
                }
            )
        )

        # Recursively scrape URLs
        self.recursive_scrape_urls(soup)

    def recursive_scrape_urls(self, soup):
        # Find all <a> tags with href attribute
        for link in soup.find_all("a", href=True):
            # Construct the absolute URL
            absolute_url = urljoin(self.base_url, link["href"])
            # Scrape the URL
            self.scrape_url(absolute_url)

    def extract_text(self, soup):
        # Extract text from HTML elements
        text = []
        for element in soup.find_all(text=True):
            if element.parent.name in ["script", "style"]:
                continue
            sentence = element.strip()
            if sentence:
                text.append(sentence + ".")
        return text

    def crawl(self, root_url, recursion=False):
        # Reset the visited URLs and data
        self.visited_urls = set()
        self.data = pl.DataFrame(
            {
                "UUID": [],
                "URL": [],
                "TEXT": [],
            }
        )

        # Set the base URL
        self.base_url = root_url

        # Scrape the root URL
        self.scrape_url(root_url)

        # Recursively scrape URLs if enabled
        if recursion:
            self.recursive_scrape_urls(BeautifulSoup(
                requests.get(root_url).content, "html.parser"))


# Example usage
crawler = GeneralWebCrawler()
crawler.crawl("https://example.com", recursion=True)

# Print the data
print(crawler.data)
