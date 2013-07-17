from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.selector import HtmlXPathSelector
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import HtmlXPathSelector
from scrapy.item import Item, Field
import re

snpp_base_url = "http://www.snpp.com/episodes.html"
ss_base_url = "http://www.springfieldspringfield.co.uk/episode_scripts.php"
sc_base_url = "http://www.simpsoncrazy.com/scripts"

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

class SimpsonsSpider2(CrawlSpider):
    name = "sc"
    allowed_domains = ['www.simpsoncrazy.com', 'simpsoncrazy.com']
    start_urls = [sc_base_url]
    rules = [Rule(SgmlLinkExtractor(allow=['/scripts/\w+']), 'parse_script')]

    def fix_field_names(self, field_name):
        field_name = re.sub(" ","_", field_name)
        field_name = re.sub(":","", field_name)
        return field_name

    def parse_script(self, response):
        x = HtmlXPathSelector(response)

        script = Script()

        script['url'] = response.url
        script['episode_name'] = "".join(x.select("//h1/text()").extract())
        all_text = x.select("//p/text()").extract()


        for i in xrange(0,len(all_text)):
            all_text[i] = re.sub("[\r\t\n]","",all_text[i]).strip()

        counter = 0
        new_text = []
        while counter<len(all_text):
            if(all_text[counter].upper()==all_text[counter]):
                nt = all_text[counter].title() + ": " + all_text[counter+1]
                new_text.append(nt)
                counter=counter+2
            else:
                counter+=1

        script['script'] = "\n".join(new_text)
        return script