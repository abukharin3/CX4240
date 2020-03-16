import scrapy

class BrickSetSpider(scrapy.Spider):
    name = "brickset_spider"
    start_urls = ['https://genius.com/songs/all?page=1']#,'https://genius.com/songs/all?page=2']

    def song_parse(self, response):
        print("!!!!!!!!!!!!!!!!!!!!")
        SET_SELECTOR = 'p,lyrics ::text'
        print(response.css(SET_SELECTOR).extract_first())
        #print()

        # with open("genius.text", "a") as file:
        #     for song in response.css(SET_SELECTOR):
        #         print("\n\n\n")
        #         print(type(song))
        #         print("\n\n\n")
                #NAME_SELECTOR = 'p ::text'
                #file.writelines(song.css(NAME_SELECTOR).extract())


    def parse(self, response):
        song_list = []

        NEXT_PAGE_SELECTOR = 'a ::attr(href)'
        next_page = response.css(NEXT_PAGE_SELECTOR).extract()


        song_list = []
        for item in next_page:
            if item[-6:] == 'lyrics':
                song_list.append(item)


        print("\n\n\n")
        # Use top 50 songs
        link = song_list[0]
        yield scrapy.Request(
            response.urljoin(link),
            callback=self.song_parse,
        )
        
