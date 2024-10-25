import requests
from bs4 import BeautifulSoup as BS

HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def check_if_page_exists(URL):
    response = requests.get(URL, headers=HEADERS, timeout=10)
    print("Checking if URL: ", URL, " exists.")
    if response.status_code == 200:
        return True
    else:
        return False

def get_urls(URL):
    check_if = check_if_page_exists(URL)
    if check_if:
        response = requests.get(URL, headers=HEADERS, timeout=10)
        soup = BS(response.content, "html.parser")

        same_domain_urls = []
        potential_about_urls = []
        external_urls = []

        base_url = '/'.join(URL.split('/')[:3])

        for link in soup.find_all('a', href=True):
            href = link['href']


            if href.startswith('/'):
                absolute_url = base_url + href
            elif href.startswith('http'): # Already an absolute URL
                absolute_url = href
            else:  # Handle other relative cases or skip if needed
                absolute_url = base_url + '/' + href


            if base_url in absolute_url:
                same_domain_urls.append(absolute_url)

                about_keywords = ["about", "about-us", "know-us", "our-story", "team", "company", "knowus"]

                if any(keyword in absolute_url.lower() for keyword in about_keywords):
                    potential_about_urls.append(absolute_url)
            else:
                external_urls.append(absolute_url)
    else:
        print("CHECK URL: ", URL, ".\n Getting a 200 response error. Maybe wrong url?")


    return same_domain_urls, potential_about_urls, external_urls

def scrape_content(URL):
    check_if = check_if_page_exists(URL)
    if check_if:
        same_domain, about_domain, external_domain = get_urls(URL)
        if len(about_domain) > 0:
            print("Please wait.. Scraping content from ", about_domain[0])
            response = requests.get(about_domain[0], headers=HEADERS, timeout=10)
            soup = BS(response.content, "html.parser")

            paragraphs = soup.find_all('p')
            paragraph_text = " ".join([p.get_text(strip=True) for p in paragraphs])

            other_tags = ['span', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']
            other_text = " ".join([tag.get_text(strip=True)  for tag_name in other_tags for tag in soup.find_all(tag_name)])

            body_text = paragraph_text + " " + other_text

            return body_text
        else:
            print("Please wait.. Scraping content from ", URL)
            response = requests.get(URL, headers=HEADERS, timeout=10)
            soup = BS(response.content, "html.parser")

            paragraphs = soup.find_all('p')
            paragraph_text = " ".join([p.get_text(strip=True) for p in paragraphs])

            other_tags = ['span', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']
            other_text = " ".join([tag.get_text(strip=True)  for tag_name in other_tags for tag in soup.find_all(tag_name)])

            body_text = paragraph_text + " " + other_text

            return body_text
    else:
        print("CHECK URL: ", URL, ".\n Getting a 200 response error. Maybe wrong url?")

text_SVC = scrape_content("https://cloq.app/")
text_BERT = scrape_content("https://www.indiacitywalks.com/")
print("Cloq: ", text_SVC)
print("\n\n\n\n\nIndiacity: ", text_BERT)