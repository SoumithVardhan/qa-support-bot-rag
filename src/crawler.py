import time
import logging
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import validators

from .config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CrawledPage:
    url: str
    title: str
    content: str
    links: List[str] = field(default_factory=list)


class WebCrawler:
    """Crawls websites and extracts text content"""

    def __init__(self, base_url, max_pages=None, crawl_delay=None, allowed_domains=None):
        self.base_url = base_url.rstrip("/")
        self.max_pages = max_pages or settings.max_pages
        self.crawl_delay = crawl_delay or settings.crawl_delay

        parsed = urlparse(self.base_url)
        self.base_domain = parsed.netloc
        self.allowed_domains = set(allowed_domains or [self.base_domain])

        self.visited_urls = set()
        self.crawled_pages = []

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "RAG-QA-Bot/1.0 (Educational Project)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })

    def is_valid_url(self, url):
        if not url or not validators.url(url):
            return False

        parsed = urlparse(url)
        if parsed.netloc not in self.allowed_domains:
            return False

        # skip non-html stuff
        skip_ext = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg",
                    ".css", ".js", ".ico", ".xml", ".json", ".zip",
                    ".mp3", ".mp4", ".avi", ".mov", ".webm"}
        if any(parsed.path.lower().endswith(ext) for ext in skip_ext):
            return False

        if "#" in url:
            url = url.split("#")[0]

        return True

    def normalize_url(self, url, current_url):
        if not url:
            return None

        full_url = urljoin(current_url, url)
        if "#" in full_url:
            full_url = full_url.split("#")[0]
        full_url = full_url.rstrip("/")

        return full_url if self.is_valid_url(full_url) else None

    def extract_content(self, soup):
        # get rid of junk
        for elem in soup.find_all(["script", "style", "nav", "footer", "header",
                                   "aside", "form", "button", "noscript", "iframe"]):
            elem.decompose()

        # find the main content
        main = (soup.find("main") or
                soup.find("article") or
                soup.find(class_=["content", "main-content", "post-content"]) or
                soup.find(id=["content", "main-content", "main"]) or
                soup.body)

        if not main:
            return ""

        text = main.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return "\n".join(lines)

    def extract_title(self, soup):
        title = soup.find("title")
        if title:
            return title.get_text(strip=True)
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return "Untitled"

    def extract_links(self, soup, current_url):
        links = []
        for a in soup.find_all("a", href=True):
            normalized = self.normalize_url(a["href"], current_url)
            if normalized and normalized not in self.visited_urls:
                links.append(normalized)
        return list(set(links))

    def crawl_page(self, url):
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()

            if "text/html" not in resp.headers.get("content-type", ""):
                return None

            soup = BeautifulSoup(resp.text, "lxml")
            title = self.extract_title(soup)
            content = self.extract_content(soup)
            links = self.extract_links(soup, url)

            if len(content) < 100:
                logger.debug(f"Skipping {url} - not enough content")
                return None

            return CrawledPage(url=url, title=title, content=content, links=links)

        except requests.RequestException as e:
            logger.warning(f"Failed to crawl {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return None

    def crawl(self):
        logger.info(f"Starting crawl from {self.base_url}")
        logger.info(f"Max pages: {self.max_pages}, Delay: {self.crawl_delay}s")

        queue = [self.base_url]

        with tqdm(total=self.max_pages, desc="Crawling") as pbar:
            while queue and len(self.crawled_pages) < self.max_pages:
                url = queue.pop(0)

                if url in self.visited_urls:
                    continue
                self.visited_urls.add(url)

                page = self.crawl_page(url)
                if page:
                    self.crawled_pages.append(page)
                    pbar.update(1)
                    for link in page.links:
                        if link not in self.visited_urls:
                            queue.append(link)
                    logger.debug(f"Crawled: {url} ({len(page.content)} chars)")

                time.sleep(self.crawl_delay)

        logger.info(f"Done. Crawled {len(self.crawled_pages)} pages.")
        return self.crawled_pages

    def get_documents(self):
        return [{"content": p.content, "url": p.url, "title": p.title}
                for p in self.crawled_pages]


def crawl_website(base_url, max_pages=None):
    crawler = WebCrawler(base_url, max_pages=max_pages)
    crawler.crawl()
    return crawler.get_documents()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python crawler.py <url> [max_pages]")
        sys.exit(1)

    url = sys.argv[1]
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    docs = crawl_website(url, max_pages)
    print(f"\nCrawled {len(docs)} pages:")
    for doc in docs:
        print(f"  - {doc['title'][:50]}... ({len(doc['content'])} chars)")
