import scrapy

class BrickSetSpider(scrapy.Spider):
    name = "brickset_spider"
    start_urls = ['http://www.songlyrics.com/hip-hop-rap-lyrics.php']
    #start_urls = ['http://www.songlyrics.com/kendrick-lamar/humble-lyrics/']

    def song_parse(self, response):
        print(response)
        SET_SELECTOR = 'songLyricsDiv-outer,p'

        with open("Trial.text", "a") as file:
            for song in response.css(SET_SELECTOR):
                NAME_SELECTOR = 'p ::text'
                #print(song.css(NAME_SELECTOR).extract())
                file.writelines(song.css(NAME_SELECTOR).extract())
                # for song in song.css(NAME_SELECTOR).extract():
                #     print(count)
                #     count += 1
                #     print(song)
            # yield {
            #     'name': brickset.css(NAME_SELECTOR).extract(),
            #     }


    def parse(self, response):
        with open("Trial.text", "w") as file:
            pass
        # SET_SELECTOR = '.tracklist'
        # for brickset in response.css(SET_SELECTOR):

        #     NAME_SELECTOR = 'a ::text'
        #     yield {
        #         'name': brickset.css(NAME_SELECTOR).extract(),

        #     }


        NEXT_PAGE_SELECTOR = '.tracklist a ::attr(href)'
        next_page = response.css(NEXT_PAGE_SELECTOR).extract()

        # link = next_page[0]
        # yield scrapy.Request(
        #             response.urljoin(link),
        #             callback=self.song_parse,
        #         )

        # Use top 30 songs
        for link in next_page[0 : 60]:
            if next_page:
                yield scrapy.Request(
                    response.urljoin(link),
                    callback=self.song_parse,
                )

