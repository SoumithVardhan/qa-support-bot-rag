import time
import logging
import random
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from typing import List, Dict, Set, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from threading import Lock

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


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

    # Retryable HTTP status codes
    retryable_status_codes: tuple = (408, 429, 500, 502, 503, 504)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter"""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        if self.jitter:
            delay = delay * (0.5 + random.random())
        return delay


class RobotsChecker:
    """Handles robots.txt parsing and URL permission checking"""

    def __init__(self, user_agent: str, cache_ttl: int = 3600):
        self.user_agent = user_agent
        self.cache_ttl = cache_ttl
        self._parsers: Dict[str, RobotFileParser] = {}
        self._cache_times: Dict[str, float] = {}
        self._lock = Lock()

    def _get_robots_url(self, url: str) -> str:
        """Extract robots.txt URL from any URL"""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    def _fetch_robots(self, robots_url: str) -> Optional[RobotFileParser]:
        """Fetch and parse robots.txt file"""
        try:
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            return rp
        except Exception as e:
            logger.debug(f"Could not fetch robots.txt from {robots_url}: {e}")
            # Return a permissive parser if robots.txt is unavailable
            rp = RobotFileParser()
            rp.parse([])  # Empty rules = allow all
            return rp

    def _get_parser(self, url: str) -> RobotFileParser:
        """Get or create a robots.txt parser for the given URL's domain"""
        robots_url = self._get_robots_url(url)

        with self._lock:
            current_time = time.time()

            # Check if cached parser is still valid
            if robots_url in self._parsers:
                cache_time = self._cache_times.get(robots_url, 0)
                if current_time - cache_time < self.cache_ttl:
                    return self._parsers[robots_url]

            # Fetch new parser
            parser = self._fetch_robots(robots_url)
            self._parsers[robots_url] = parser
            self._cache_times[robots_url] = current_time
            return parser

    def can_fetch(self, url: str) -> bool:
        """Check if the URL can be fetched according to robots.txt"""
        parser = self._get_parser(url)
        return parser.can_fetch(self.user_agent, url)

    def get_crawl_delay(self, url: str) -> Optional[float]:
        """Get the crawl delay specified in robots.txt for the domain"""
        parser = self._get_parser(url)
        try:
            delay = parser.crawl_delay(self.user_agent)
            return delay
        except AttributeError:
            return None


class DomainRateLimiter:
    """Implements per-domain rate limiting for polite crawling"""

    def __init__(self, default_delay: float = 1.0, respect_robots_delay: bool = True):
        self.default_delay = default_delay
        self.respect_robots_delay = respect_robots_delay
        self._last_request: Dict[str, float] = defaultdict(float)
        self._domain_delays: Dict[str, float] = {}
        self._lock = Lock()

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        return urlparse(url).netloc

    def set_domain_delay(self, domain: str, delay: float):
        """Set a custom delay for a specific domain"""
        with self._lock:
            self._domain_delays[domain] = delay

    def get_delay_for_url(self, url: str, robots_delay: Optional[float] = None) -> float:
        """Get the appropriate delay for a URL's domain"""
        domain = self._get_domain(url)

        # Priority: custom domain delay > robots.txt delay > default delay
        if domain in self._domain_delays:
            return self._domain_delays[domain]

        if self.respect_robots_delay and robots_delay is not None:
            return max(robots_delay, self.default_delay)

        return self.default_delay

    def wait_if_needed(self, url: str, robots_delay: Optional[float] = None):
        """Wait if necessary before making a request to respect rate limits"""
        domain = self._get_domain(url)
        required_delay = self.get_delay_for_url(url, robots_delay)

        with self._lock:
            last_request_time = self._last_request[domain]
            current_time = time.time()
            elapsed = current_time - last_request_time

            if elapsed < required_delay:
                wait_time = required_delay - elapsed
                time.sleep(wait_time)

            self._last_request[domain] = time.time()


class WebCrawler:
    """
    Production-ready web crawler with:
    - robots.txt compliance
    - Exponential backoff retry logic
    - Per-domain rate limiting (politeness)
    - Configurable retry strategies
    """

    DEFAULT_USER_AGENT = "RAG-QA-Bot/1.0 (Educational Project; Respectful Crawler)"

    def __init__(
        self,
        base_url: str,
        max_pages: Optional[int] = None,
        crawl_delay: Optional[float] = None,
        allowed_domains: Optional[List[str]] = None,
        retry_config: Optional[RetryConfig] = None,
        respect_robots: bool = True,
        user_agent: Optional[str] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.max_pages = max_pages or settings.max_pages
        self.crawl_delay = crawl_delay or settings.crawl_delay
        self.respect_robots = respect_robots
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT

        parsed = urlparse(self.base_url)
        self.base_domain = parsed.netloc
        self.allowed_domains = set(allowed_domains or [self.base_domain])

        self.visited_urls: Set[str] = set()
        self.failed_urls: Dict[str, str] = {}  # URL -> error message
        self.crawled_pages: List[CrawledPage] = []

        # Initialize retry configuration
        self.retry_config = retry_config or RetryConfig(
            max_retries=settings.max_retries,
            base_delay=settings.retry_base_delay,
            max_delay=settings.retry_max_delay,
            exponential_base=settings.retry_exponential_base,
            jitter=settings.retry_jitter
        )

        # Initialize robots.txt checker
        self.robots_checker = RobotsChecker(
            user_agent=self.user_agent,
            cache_ttl=settings.robots_cache_ttl
        ) if self.respect_robots else None

        # Initialize rate limiter
        self.rate_limiter = DomainRateLimiter(
            default_delay=self.crawl_delay,
            respect_robots_delay=self.respect_robots
        )

        # Setup session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and allowed to be crawled"""
        if not url or not validators.url(url):
            return False

        parsed = urlparse(url)
        if parsed.netloc not in self.allowed_domains:
            return False

        # Skip non-HTML resources
        skip_ext = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg",
                    ".css", ".js", ".ico", ".xml", ".json", ".zip",
                    ".mp3", ".mp4", ".avi", ".mov", ".webm", ".woff",
                    ".woff2", ".ttf", ".eot"}
        if any(parsed.path.lower().endswith(ext) for ext in skip_ext):
            return False

        return True

    def is_allowed_by_robots(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt"""
        if not self.respect_robots or self.robots_checker is None:
            return True
        return self.robots_checker.can_fetch(url)

    def normalize_url(self, url: str, current_url: str) -> Optional[str]:
        """Normalize and validate a URL"""
        if not url:
            return None

        full_url = urljoin(current_url, url)

        # Remove fragment
        if "#" in full_url:
            full_url = full_url.split("#")[0]

        # Remove trailing slash for consistency
        full_url = full_url.rstrip("/")

        if not self.is_valid_url(full_url):
            return None

        return full_url

    def extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from page"""
        # Remove non-content elements
        for elem in soup.find_all(["script", "style", "nav", "footer", "header",
                                   "aside", "form", "button", "noscript", "iframe",
                                   "meta", "link"]):
            elem.decompose()

        # Find main content area
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

    def extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title = soup.find("title")
        if title:
            return title.get_text(strip=True)
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return "Untitled"

    def extract_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """Extract and normalize all links from page"""
        links = []
        for a in soup.find_all("a", href=True):
            normalized = self.normalize_url(a["href"], current_url)
            if normalized and normalized not in self.visited_urls:
                links.append(normalized)
        return list(set(links))

    def _make_request_with_retry(self, url: str) -> Optional[requests.Response]:
        """
        Make HTTP request with exponential backoff retry logic.

        Handles transient network failures and rate limiting.
        """
        last_exception = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                response = self.session.get(url, timeout=10)

                # Check for retryable status codes
                if response.status_code in self.retry_config.retryable_status_codes:
                    if attempt < self.retry_config.max_retries:
                        delay = self.retry_config.get_delay(attempt)

                        # Check for Retry-After header
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                delay = max(delay, float(retry_after))
                            except ValueError:
                                pass

                        logger.warning(
                            f"Retryable status {response.status_code} for {url}. "
                            f"Attempt {attempt + 1}/{self.retry_config.max_retries + 1}. "
                            f"Waiting {delay:.2f}s"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Max retries exceeded for {url} (status: {response.status_code})")
                        return None

                response.raise_for_status()
                return response

            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt)
                    logger.warning(
                        f"Timeout for {url}. Attempt {attempt + 1}/{self.retry_config.max_retries + 1}. "
                        f"Waiting {delay:.2f}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Max retries exceeded for {url} (timeout)")

            except requests.exceptions.ConnectionError as e:
                last_exception = e
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt)
                    logger.warning(
                        f"Connection error for {url}. Attempt {attempt + 1}/{self.retry_config.max_retries + 1}. "
                        f"Waiting {delay:.2f}s"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Max retries exceeded for {url} (connection error)")

            except requests.exceptions.HTTPError as e:
                # Non-retryable HTTP errors (4xx except 408, 429)
                logger.warning(f"HTTP error for {url}: {e}")
                return None

            except requests.RequestException as e:
                last_exception = e
                logger.warning(f"Request failed for {url}: {e}")
                return None

        if last_exception:
            self.failed_urls[url] = str(last_exception)
        return None

    def crawl_page(self, url: str) -> Optional[CrawledPage]:
        """Crawl a single page with all safety checks"""
        # Check robots.txt permission
        if not self.is_allowed_by_robots(url):
            logger.debug(f"Blocked by robots.txt: {url}")
            return None

        # Get robots.txt crawl delay for this domain
        robots_delay = None
        if self.robots_checker:
            robots_delay = self.robots_checker.get_crawl_delay(url)

        # Apply rate limiting
        self.rate_limiter.wait_if_needed(url, robots_delay)

        # Make request with retry logic
        response = self._make_request_with_retry(url)
        if response is None:
            return None

        # Verify content type
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            return None

        try:
            soup = BeautifulSoup(response.text, "lxml")
            title = self.extract_title(soup)
            content = self.extract_content(soup)
            links = self.extract_links(soup, url)

            if len(content) < 100:
                logger.debug(f"Skipping {url} - insufficient content")
                return None

            return CrawledPage(url=url, title=title, content=content, links=links)

        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            self.failed_urls[url] = str(e)
            return None

    def crawl(self) -> List[CrawledPage]:
        """
        Execute the crawl with all production-ready features:
        - robots.txt compliance
        - Retry with exponential backoff
        - Per-domain rate limiting
        """
        logger.info(f"Starting crawl from {self.base_url}")
        logger.info(f"Max pages: {self.max_pages}, Base delay: {self.crawl_delay}s")
        logger.info(f"Robots.txt compliance: {'enabled' if self.respect_robots else 'disabled'}")
        logger.info(f"Retry config: max_retries={self.retry_config.max_retries}, "
                   f"base_delay={self.retry_config.base_delay}s")

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
                        if link not in self.visited_urls and link not in queue:
                            queue.append(link)

                    logger.debug(f"Crawled: {url} ({len(page.content)} chars, {len(page.links)} links)")

        # Log summary
        logger.info(f"Crawl complete. Successfully crawled {len(self.crawled_pages)} pages.")
        if self.failed_urls:
            logger.warning(f"Failed to crawl {len(self.failed_urls)} URLs")

        return self.crawled_pages

    def get_documents(self) -> List[Dict[str, str]]:
        """Get crawled pages as document dictionaries"""
        return [{"content": p.content, "url": p.url, "title": p.title}
                for p in self.crawled_pages]

    def get_stats(self) -> Dict:
        """Get crawling statistics"""
        return {
            "pages_crawled": len(self.crawled_pages),
            "pages_visited": len(self.visited_urls),
            "pages_failed": len(self.failed_urls),
            "failed_urls": dict(self.failed_urls)
        }


def crawl_website(
    base_url: str,
    max_pages: Optional[int] = None,
    respect_robots: bool = True,
    retry_config: Optional[RetryConfig] = None
) -> List[Dict[str, str]]:
    """
    Convenience function to crawl a website.

    Args:
        base_url: Starting URL for the crawl
        max_pages: Maximum number of pages to crawl
        respect_robots: Whether to respect robots.txt rules
        retry_config: Custom retry configuration

    Returns:
        List of document dictionaries with content, url, and title
    """
    crawler = WebCrawler(
        base_url,
        max_pages=max_pages,
        respect_robots=respect_robots,
        retry_config=retry_config
    )
    crawler.crawl()
    return crawler.get_documents()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python crawler.py <url> [max_pages]")
        sys.exit(1)

    url = sys.argv[1]
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print(f"Crawling {url} with production-ready features...")
    print("- robots.txt compliance: enabled")
    print("- Exponential backoff: enabled")
    print("- Per-domain rate limiting: enabled")
    print()

    docs = crawl_website(url, max_pages, respect_robots=True)

    print(f"\nCrawled {len(docs)} pages:")
    for doc in docs:
        print(f"  - {doc['title'][:50]}... ({len(doc['content'])} chars)")
