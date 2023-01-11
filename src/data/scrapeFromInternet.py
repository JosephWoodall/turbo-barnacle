import requests
from bs4 import BeautifulSoup

class WebScraper:
    def __init__(self, url):
        self.url = url

    def get_data(self):
        # Send a GET request to the website
        response = requests.get(self.url)

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        return soup
    
    def extract_data(self, tag, attribute, value):
        # Extract data based on the specified tag, attribute, and value
        data = []
        soup = self.get_data()
        for element in soup.find_all(tag, {attribute: value}):
            data.append(element.text)
        return data