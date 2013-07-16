from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.selector import HtmlXPathSelector
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import HtmlXPathSelector
from scrapy.item import Item, Field
import re

base_url = "http://www.snpp.com/episodes.html"

class Script(Item):
    url = Field()
    episode_name = Field()
    script = Field()

class SimpsonsSpider(CrawlSpider):
    name = "snpp"
    allowed_domains = ['www.snpp.com', 'snpp.com']
    start_urls = [base_url]
    rules = [Rule(SgmlLinkExtractor(allow=['/episodes/\w+.html']), 'parse_script')]

    def fix_field_names(self, field_name):
        field_name = re.sub(" ","_", field_name)
        field_name = re.sub(":","", field_name)
        return field_name

    def parse_script(self, response):
        x = HtmlXPathSelector(response)

        script = Script()

        script['url'] = response.url
        script['episode_name'] = "".join(x.select("//h1/text()").extract())
        script['script'] = " ".join(x.select("//pre/text()").extract())
        return script