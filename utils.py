import requests
from bs4 import BeautifulSoup

def get_link_preview(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    def get_meta(property_name):
        tag = soup.find("meta", property=property_name)
        if tag:
            return tag.get("content")
        return None

    title = get_meta("og:title") or soup.title.string if soup.title else ""
    description = get_meta("og:description")
    image = get_meta("og:image")

    return {
        "title": title,
        "description": description,
        "image": image,
        "url": url
    }
