import scrapy

class BrickSetSpider(scrapy.Spider):
    name = "brickset_spider"
    start_urls = ['https://genius.com/tags/r-b/all?page=' + str(i) for i in range(1, 51)]#,'https://genius.com/songs/all?page=2']

    def song_parse(self, response):
        SET_SELECTOR = '.lyrics'
        with open("rnb_songs.text", "a") as file:
            for song in response.css(SET_SELECTOR):
                NAME_SELECTOR = 'p ::text'
                file.writelines(["\n\n\nStart of song \n\n\n"])
                file.writelines(song.css(NAME_SELECTOR).extract())



    def parse(self, response):
        with open("rnb_songs.text", "a") as file:
            pass

        song_list = []

        NEXT_PAGE_SELECTOR = 'a ::attr(href)'
        next_page = response.css(NEXT_PAGE_SELECTOR).extract()

        song_list = []
        for item in next_page:
            if item[-6:] == 'lyrics':
                print(item)
                song_list.append(item)


        for link in song_list:
            if song_list:
                yield scrapy.Request(
                    response.urljoin(link),
                    callback=self.song_parse,
                )
