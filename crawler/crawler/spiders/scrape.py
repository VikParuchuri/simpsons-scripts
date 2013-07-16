from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.selector import HtmlXPathSelector
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import HtmlXPathSelector
from scrapy.item import Item, Field
import re

snpp_base_url = "http://www.snpp.com/episodes.html"
ss_base_url = "http://www.springfieldspringfield.co.uk/episode_scripts.php"

class Script(Item):
    url = Field()
    episode_name = Field()
    script = Field()

class SimpsonsSpider(CrawlSpider):
    name = "snpp"
    allowed_domains = ['www.snpp.com', 'snpp.com']
    start_urls = [snpp_base_url]
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

class SubtitleSpider(CrawlSpider):
    name = "ss"
    allowed_domains = ['www.springfieldspringfield.co.uk', 'springfieldspringfield.co.uk']
    start_urls = [ss_base_url]
    rules = [Rule(SgmlLinkExtractor(allow=['/view_episode_scripts.php\?tv-show=the-simpsons&episode=\w+']), 'parse_script')]

    def fix_field_names(self, field_name):
        field_name = re.sub(" ","_", field_name)
        field_name = re.sub(":","", field_name)
        return field_name

    def parse_script(self, response):
        x = HtmlXPathSelector(response)

        script = Script()

        script['url'] = response.url
        script['episode_name'] = "".join(x.select("//h3/text()").extract())
        script['script'] = "\n".join(x.select("//div[@class='episode_script']/text()").extract())
        return script