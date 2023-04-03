import requests
from bs4 import BeautifulSoup
import itertools
import json
import urllib.request


class InternetIndexer:
    """
    This class is a first pass at creating a comprehensive list of all 
    publicly accessible websites, which, would require a large amount of 
    resources and computing power; so this is a PoC.
    """

    def __init__(self, max_records=2):
        self.max_records = max_records

    def crawl_web(self, url):
        """_summary_

        Args:
            url (_type_): _description_

        Returns:
            _type_: _description_
        """
        visited = set()
        queue = [url]
        while queue and len(visited) < self.max_records:
            url = queue.pop(0)
            if url not in visited:
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        # Process the page content and extract URLs
                        for link in soup.find_all('a'):
                            href = link.get('href')
                            if href and href.startswith('http'):
                                queue.append(href)
                except:
                    pass
                visited.add(url)
        return visited

    def common_crawl(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        url = 'http://index.commoncrawl.org/collinfo.json'
        response = urllib.request.urlopen(url)
        data = json.loads(response.read())

        urls = []
        for item in data:
            if 'cdx-api' in item:
                cdx_api = item['cdx-api']
                url = '{}?url={}&output=json'.format(cdx_api, 'org', 'com')
                response = urllib.request.urlopen(url)
                lines = response.read().decode('utf-8').split('\n')
                for line in lines:
                    if line and len(urls) < self.max_records:
                        fields = line.split(' ')
                        urls.append(fields[2])
        return urls

    def domain_generator(self, length, keywords):
        """_summary_

        Args:
            length (_type_): _description_
            keywords (_type_): _description_

        Yields:
            _type_: _description_
        """
        vowels = ['a', 'e', 'i', 'o', 'u']
        consonants = list(set('abcdefghijklmnopqrstuvwxyz') - set(vowels))
        count = 0
        for tld in itertools.product(consonants, repeat=3):
            tld_str = ''.join(tld)
            for domain in itertools.product(keywords, vowels, consonants, repeat=length - 4):
                domain_str = ''.join(domain)
                if count >= self.max_records:
                    return
                yield domain_str + tld_str
                count += 1

    def generate_websites(self, method='crawl_web', seed_url=None, length=None, keywords=None):
        """_summary_

        Args:
            method (str, optional): _description_. Defaults to 'crawl_web'.
            seed_url (_type_, optional): _description_. Defaults to None.
            length (_type_, optional): _description_. Defaults to None.
            keywords (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if method == 'crawl_web':
            return self.crawl_web(seed_url)
        elif method == 'common_crawl':
            return self.common_crawl()
        elif method == 'domain_generator':
            return list(self.domain_generator(length, keywords))
        else:
            raise ValueError('Invalid method name')


indexer = InternetIndexer(max_records=2)

# Generate list of websites using web crawling method
websites = indexer.generate_websites(
    method='crawl_web', seed_url='https://www.google.com')
print(len(websites))

"""
# Generate list of websites using Common Crawl dataset
websites = indexer.generate_websites(method='common_crawl')
print(len(websites))

# Generate list of websites using domain generator
websites = indexer.generate_websites(
    method='domain_generator', length=8, keywords=['google', 'apple', 'amazon'])
print(len(websites))
"""
