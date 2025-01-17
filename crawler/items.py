# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from dataclasses import dataclass


@dataclass
class PhoneCrawlerItem:
    listing_id: str
    url: str
    content: dict
    crawl_date: str
    source: str
