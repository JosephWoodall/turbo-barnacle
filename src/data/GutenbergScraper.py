import requests
import json
import re

class GutenbergScraper:
    def __init__(self, num_books):
        self.num_books = num_books
        self.base_url = f'https://www.gutenberg.org/cache/epub/{{}}/pg{{}}.txt'

    def scrape_book_text(self, url):
        response = requests.get(url)
        book_text = response.text
        cleaned_text = self.remove_special_characters(book_text)
        return cleaned_text

    def remove_special_characters(self, text):
        # Define a regex pattern to match non-alphanumeric characters and whitespace
        pattern = r'[^a-zA-Z0-9\s]'

        # Remove the special characters from the text
        cleaned_text = re.sub(pattern, '', text)

        # Remove newlines and carriage returns
        cleaned_text = cleaned_text.replace('\n', '').replace('\r', '')
        
        return cleaned_text

    def scrape_books(self):
        books = []

        for book_id in range(1, self.num_books + 1):
            book_url = self.base_url.format(book_id, book_id)
            book_text = self.scrape_book_text(book_url)

            book_data = {
                'book_id': book_id,
                'book_url': book_url,
                'book_text': book_text
            }

            books.append(book_data)

        with open(r'src/data/scraped_books.json', 'w') as file:
            json.dump(books, file, indent=4)

        print('Scraping completed. Data saved to scraped_books.json')
        
# Usage
num_books_to_scrape = 10
scraper = GutenbergScraper(num_books_to_scrape)
scraper.scrape_books()
