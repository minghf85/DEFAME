"""A web scraping module to retrieve the contents of ANY website."""
import asyncio
from typing import Optional

import aiohttp
import requests
from ezmm import MultimodalSequence, Image
from firecrawl import AsyncFirecrawlApp

from config.globals import firecrawl_url,firecrawl_api_key
from defame.common import logger
from defame.evidence_retrieval.integrations import RETRIEVAL_INTEGRATIONS
from defame.evidence_retrieval.integrations.search import WebSource
from defame.evidence_retrieval.scraping.excluded import (is_unsupported_site, is_relevant_content,
                                                         is_fact_checking_site)
from defame.evidence_retrieval.scraping.util import scrape_naive, find_firecrawl, firecrawl_is_running, log_error_url, \
    resolve_media_hyperlinks
from defame.utils.parsing import get_domain
from defame.utils.requests import download, is_image_url

# ...existing code...


class Scraper:
    """Takes any URL and tries to scrape its contents. If the URL belongs to a platform
    requiring an API and the API integration is implemented (e.g. X, Reddit etc.), the
    respective API will be used instead of direct HTTP requests."""

    def __init__(self, allow_fact_checking_sites: bool = True):
        self.allow_fact_checking_sites = allow_fact_checking_sites
        self.firecrawl_app = AsyncFirecrawlApp(api_key=firecrawl_api_key)
        self.n_scrapes = 0
        logger.log(f"✅ Initialized cloud Firecrawl with API key.")

    def locate_firecrawl(self):
        """Cloud Firecrawl doesn't require location detection."""
        pass

    def scrape_sources(self, sources: list[WebSource]) -> None:
        """Retrieves the contents for the given web sources and saves them
        into the respective web source object."""
        # Only keep sources that weren't scraped yet
        sources = [s for s in sources if not s.is_loaded()]

        if sources:
            urls = [s.url for s in sources]
            scrape_results = self.scrape_multiple(urls)
            for source, scraped in zip(sources, scrape_results):
                source.content = scraped

    def scrape_multiple(self, urls: list[str]) -> list[MultimodalSequence | None]:
        """Scrapes each URL concurrently. Synchronous wrapper for _scrape_multiple()."""
        return asyncio.run(self._scrape_multiple(urls))

    def scrape(self, url: str) -> Optional[MultimodalSequence]:
        """Scrapes the contents of the specified webpage. Synchronous wrapper for _scrape()."""
        return asyncio.run(self._scrape(url))

    async def _scrape_multiple(self, urls: list[str]) -> list[MultimodalSequence | None]:
        tasks = [self._scrape(url) for url in urls]
        return await asyncio.gather(*tasks)

    async def _scrape(self, url: str) -> Optional[MultimodalSequence]:
        # Check exclusions first
        if is_unsupported_site(url):
            logger.log(f"Skipping unsupported site: {url}")
            return None
        if not self.allow_fact_checking_sites and is_fact_checking_site(url):
            logger.log(f"Skipping fact-checking site: {url}")
            return None

        # Identify and use any applicable integration to retrieve the URL contents
        scraped = _retrieve_via_integration(url)
        if scraped:
            return scraped

        # Check if URL points to a media file. If yes, download accordingly TODO: extend to videos/audios
        if is_image_url(url):
            try:
                image = Image(binary_data=download(url))
                scraped = MultimodalSequence([image])
            except Exception:
                pass

        # Use cloud Firecrawl to scrape from the URL
        if not scraped:
            scraped = await self._scrape_firecrawl(url)

        # If the scrape still was not successful, use naive Beautiful Soup scraper
        if not scraped:
            scraped = scrape_naive(url)

        if scraped:
            self.n_scrapes += 1
            return scraped

    async def _scrape_firecrawl(self, url: str) -> Optional[MultimodalSequence]:
        """Scrapes the given URL using cloud Firecrawl. Returns a Markdown-formatted
        multimedia snippet, containing any (relevant) media from the page."""
        
        try:
            response = await self.firecrawl_app.scrape_url(
                url=url,
                formats=['markdown'],
                only_main_content=True,
                parse_pdf=True
            )
            
            if response and 'markdown' in response:
                text = response['markdown']
                return resolve_media_hyperlinks(text)
            else:
                error_message = f"Unable to read {url}. No markdown data in response."
                logger.info(f"Unable to read {url}. Skipping it.")
                logger.info(str(response))
                log_error_url(url, error_message)
                return None
                
        except Exception as e:
            error_message = f"Exception: {repr(e)}"
            logger.info(repr(e))
            logger.info(f"Unable to scrape {url} with cloud Firecrawl. Skipping...")
            log_error_url(url, error_message)
            return None


def _retrieve_via_integration(url: str) -> Optional[MultimodalSequence]:
    domain = get_domain(url)
    if domain in RETRIEVAL_INTEGRATIONS:
        integration = RETRIEVAL_INTEGRATIONS[domain]
        return integration.retrieve(url)


scraper = Scraper()

if __name__ == "__main__":
    print(scrape_naive("https://www.washingtonpost.com/video/national/cruz-calls-trump-clinton-two-new-york-liberals/2016/04/07/da3b78a8-fcdf-11e5-813a-90ab563f0dde_video.html"))
    # print(scraper.scrape("https://news.sina.com.cn/c/xl/2025-07-15/doc-inffqiif5367434.shtml"))
    # print(scraper.scrape_multiple([
    #     "https://www.washingtonpost.com/video/national/cruz-calls-trump-clinton-two-new-york-liberals/2016/04/07/da3b78a8-fcdf-11e5-813a-90ab563f0dde_video.html",
    #     "https://cdn.pixabay.com/photo/2017/11/08/22/28/camera-2931883_1280.jpg",
    #     "https://www.tagesschau.de/ausland/asien/libanon-israel-blauhelme-nahost-102.html",
    #     "https://www.zeit.de/politik/ausland/2024-10/wolodymyr-selenskyj-berlin-olaf-scholz-militaerhilfe",
    #     "https://pixum-cms.imgix.net/7wL8j3wldZEONCSZB9Up6B/d033b7b6280687ce2e4dfe2d4147ff93/fab_mix_kv_perspektive_foto_liegend_desktop__3_.png?auto=compress,format&trim=false&w=2000",
    # ]))
