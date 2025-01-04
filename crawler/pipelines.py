# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
# from itemadapter import ItemAdapter

from datetime import datetime
from crawler.items import PhoneCrawlerItem
from loguru import logger
from pathlib import Path
import json
from dataclasses import asdict


class PhoneCrawlerPipeline:
    def __init__(self):
        date = datetime.now()
        self.current_date = date.strftime(r"%Y-%m-%d")
        self.total_posts = 0

    def open_spider(self, spider):
        logger.info("Spider started pipeline")
        self.save_path = Path(f"data/{self.current_date}.jsonl")

    def process_item(self, item: PhoneCrawlerItem, spider):
        with open(self.save_path, "a") as f:
            f.write(json.dumps(asdict(item)) + "\n")
        self.total_posts += 1

    def close_spider(self, spider):
        logger.info(f"Spider finished pipeline. Saved {self.total_posts} posts")
        logger.info(f"Saved to {self.save_path}")
