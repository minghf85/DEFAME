import os
import time
from dataclasses import dataclass
from typing import Sequence
import urllib.request

from ezmm import Image
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import vision
from google.api_core.exceptions import ServiceUnavailable, DeadlineExceeded
import grpc

from config.globals import google_service_account_key_path
from defame.common import logger
from defame.evidence_retrieval.integrations.search.common import WebSource, Query, SearchMode, SearchResults
from defame.utils.parsing import get_base_domain


@dataclass
class GoogleRisResults(SearchResults):
    """Reverse Image Search (RIS) results. Ship with additional object detection
    information next to the list of sources."""
    entities: dict[str, float]  # mapping between entity description and confidence score
    best_guess_labels: list[str]

    @property
    def exact_matches(self):
        return self.sources

    def __str__(self):
        text = "**Reverse Image Search Results**"

        if self.entities:
            text += f"\n\nIdentified entities (confidence in parenthesis):\n"
            text += "\n".join(f"- {name} ({confidence * 100:.0f} %)"
                              for name, confidence in self.entities.items())

        if self.best_guess_labels:
            text += f"\n\nBest guess about the topic of the image: {', '.join(self.best_guess_labels)}."

        if self.exact_matches:
            text += "\n\nThe same image was found in the following sources:\n"
            text += "\n".join(map(str, self.exact_matches))

        return text

    def __repr__(self):
        return (f"RisResults(n_exact_matches={len(self.exact_matches)}, "
                f"n_entities={len(self.entities)}, "
                f"n_best_guess_labels={len(self.best_guess_labels)})")


class GoogleVisionAPI:
    """Wraps the Google Cloud Vision API for performing reverse image search (RIS)."""

    def __init__(self):
        # 设置代理环境变量
        self._setup_proxy()
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_service_account_key_path.as_posix()
        try:
            # 配置gRPC通道选项以支持代理
            channel_options = [
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.http2.min_time_between_pings_ms', 10000),
                ('grpc.http2.min_ping_interval_without_data_ms', 300000)
            ]
            
            self.client = vision.ImageAnnotatorClient(
                client_options={"api_endpoint": "vision.googleapis.com:443"}
            )
        except DefaultCredentialsError:
            logger.warning(f"❌ No or invalid Google Cloud API credentials at "
                           f"{google_service_account_key_path.as_posix()}.")
        else:
            logger.log(f"✅ Successfully connected to Google Cloud Vision API.")

    def _setup_proxy(self):
        """检测并设置代理配置"""
        # 检查常见的代理端口
        proxy_ports = [7890, 8080, 7899, 1080]
        proxy_host = "127.0.0.1"
        
        working_proxy = None
        for port in proxy_ports:
            try:
                # 测试代理连接
                proxy_url = f"http://{proxy_host}:{port}"
                proxy_handler = urllib.request.ProxyHandler({
                    'http': proxy_url,
                    'https': proxy_url
                })
                opener = urllib.request.build_opener(proxy_handler)
                urllib.request.install_opener(opener)
                
                # 简单测试代理是否工作
                req = urllib.request.Request('https://www.google.com', headers={'User-Agent': 'Mozilla/5.0'})
                response = urllib.request.urlopen(req, timeout=5)
                if response.status == 200:
                    working_proxy = proxy_url
                    break
            except:
                continue
        
        if working_proxy:
            logger.log(f"✅ 检测到可用代理: {working_proxy}")
            # 设置环境变量
            os.environ['HTTP_PROXY'] = working_proxy
            os.environ['HTTPS_PROXY'] = working_proxy
            os.environ['http_proxy'] = working_proxy
            os.environ['https_proxy'] = working_proxy
        else:
            logger.warning("❌ 未检测到可用的代理服务器")

    def search(self, query: Query, max_retries: int = 3) -> GoogleRisResults:
        """Run image reverse search through Google Vision API and parse results."""
        assert query.has_image(), "Google Vision API requires an image in the query."

        image = vision.Image(content=query.image.get_base64_encoded())
        
        last_error = None
        for attempt in range(max_retries):
            try:
                logger.log(f"尝试连接Google Vision API (第 {attempt + 1}/{max_retries} 次)")
                response = self.client.web_detection(
                    image=image,
                    timeout=30.0  # 增加超时时间
                )
                
                if response.error.message:
                    logger.warning(f"{response.error.message}\nCheck Google Cloud Vision API documentation for more info.")
                    
                return _parse_results(response.web_detection, query)
                
            except (ServiceUnavailable, DeadlineExceeded, grpc.RpcError) as e:
                last_error = e
                logger.warning(f"❌ 连接失败 (第 {attempt + 1} 次): {str(e)}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.log(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    
                    # 重新设置代理
                    self._setup_proxy()
                else:
                    logger.error("❌ 所有重试都失败了")
                    
        # 如果所有重试都失败，返回空结果而不是抛出异常
        logger.error(f"❌ 无法连接到Google Vision API: {last_error}")
        return GoogleRisResults(
            sources=[],
            query=query,
            entities={},
            best_guess_labels=[]
        )


google_vision_api = GoogleVisionAPI()


def _parse_results(web_detection: vision.WebDetection, query: Query) -> GoogleRisResults:
    """Parse Google Vision API web detection results into SearchResult instances."""

    # Web Entities
    web_entities = {}
    for entity in web_detection.web_entities:
        if entity.description:
            web_entities[entity.description] = entity.score

    # Best Guess Labels
    best_guess_labels = []
    if web_detection.best_guess_labels:
        for label in web_detection.best_guess_labels:
            if label.label:
                best_guess_labels.append(label.label)

    # Pages with relevant images
    web_sources = []
    filtered_pages = _filter_unique_stem_pages(web_detection.pages_with_matching_images)
    for page in filtered_pages:
        url = page.url
        title = page.__dict__.get("page_title")
        web_source = WebSource(reference=url, title=title)
        web_sources.append(web_source)

    return GoogleRisResults(sources=web_sources,
                            query=query,
                            entities=web_entities,
                            best_guess_labels=best_guess_labels)


def _filter_unique_stem_pages(pages: Sequence):
    """
    Filters pages to ensure only one page per website base domain is included 
    (e.g., 'facebook.com' regardless of subdomain), 
    and limits the total number of pages to the specified limit.
    
    Args:
        pages (list): List of pages with matching images.
    
    Returns:
        list: Filtered list of pages.
    """
    unique_domains = set()
    filtered_pages = []

    for page in pages:
        base_domain = get_base_domain(page.url)

        # Check if we already have a page from this base domain
        if base_domain not in unique_domains:
            unique_domains.add(base_domain)
            filtered_pages.append(page)

    return filtered_pages


if __name__ == "__main__":
    example_query = Query(
        image=Image("in/example/sahara.webp"),
        search_mode=SearchMode.REVERSE,
    )
    api = GoogleVisionAPI()
    result = api.search(example_query)
    print(result)
