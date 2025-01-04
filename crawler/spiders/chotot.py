import scrapy
from crawler.settings import NUM_PAGES
from loguru import logger
import json
from crawler.items import PhoneCrawlerItem
from datetime import datetime


class UsedSmartphoneChototSpier(scrapy.Spider):
    name = "chotot"
    allowed_domains = ["chotot.com"]
    start_urls = [
        f"https://www.chotot.com/mua-ban-dien-thoai?page={i}" for i in range(NUM_PAGES)
    ]
    crawl_date = datetime.now().strftime(r"%Y-%m-%d")

    def parse(self, response: scrapy.http.Response):
        listings = json.loads(
            response.xpath('//script[@id="__NEXT_DATA__"]/text()').get()
        )
        urls = [
            f"{listing['list_id']}.htm"
            for listing in listings["props"]["pageProps"]["initialState"]["adlisting"][
                "data"
            ]["ads"]
        ]
        logger.info(f"Found {len(urls)} listings @ {response.url}")
        yield from response.follow_all(urls, self.parse_post)

    @logger.catch
    def parse_post(self, response):
        json_content = response.xpath('//script[@id="__NEXT_DATA__"]/text()').get()
        json_content = json.loads(json_content)
        results = {}
        results["listing_id"] = json_content["query"]["listId"]
        results["content"] = {}
        results["url"] = json_content["props"]["canonicalUrl"]
        results["content"]["props"] = {}
        results["content"]["props"]["pageProps"] = {}
        results["content"]["props"]["pageProps"]["initialState"] = {}
        results["content"]["props"]["pageProps"]["initialState"]["adView"] = {}
        results["content"]["props"]["pageProps"]["initialState"]["adView"][
            "adInfo"
        ] = {}
        results["content"]["props"]["pageProps"]["initialState"]["adView"]["adInfo"][
            "ad"
        ] = json_content["props"]["pageProps"]["initialState"]["adView"]["adInfo"]["ad"]
        results["content"]["props"]["pageProps"]["initialState"]["adView"]["adInfo"][
            "ad_params"
        ] = json_content["props"]["pageProps"]["initialState"]["adView"]["adInfo"][
            "ad_params"
        ]
        results["content"]["props"]["pageProps"]["initialState"]["nav"] = {}
        results["content"]["props"]["pageProps"]["initialState"]["nav"]["navObj"] = (
            json_content["props"]["initialState"]["nav"]["navObj"]
        )
        yield PhoneCrawlerItem(
            listing_id=results["listing_id"],
            url=results["url"],
            content=results["content"],
            crawl_date=self.crawl_date,
            source="chotot",
        )
