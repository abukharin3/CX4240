import scrapy

class BrickSetSpider(scrapy.Spider):
    name = "brickset_spider"
    start_urls = ['https://genius.com/songs/all?page=1']#,'https://genius.com/songs/all?page=2']

    def song_parse(self, response):
        print(response)
        SET_SELECTOR = 'p,lyrics ::text'
        print(response.css(SET_SELECTOR).extract_first())

                

    def parse(self, response):
        with open("genius.text", "w") as file:
            pass
        
        song_list = []

        NEXT_PAGE_SELECTOR = 'a ::attr(href)'
        next_page = response.css(NEXT_PAGE_SELECTOR).extract()
        print("\n\n\n")
        #print(next_page)

        song_list = []
        for item in next_page:
            if item[-6:] == 'lyrics':
                print(item)
                song_list.append(item)


        print("\n\n\n")
        # Use top 50 songs
        
        for link in song_list[0 : 1]:
            if song_list:
                yield scrapy.Request(
                    response.urljoin(link),
                    callback=self.song_parse,
                )
        
