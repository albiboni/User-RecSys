from selenium import webdriver

class Crawler(object):

    def __init__(self, driver, mode):
        '''driver= path of chrome_driver, mode='attributes' or description or both'''
        self.driver = driver
        self.mode = mode

    def load_driver(self, url):

        drive = webdriver.Chrome(self.driver)
        drive.get(url)
        return drive

    def write_file(self, filename):

        f = open(filename, 'w')
        return f

    def get_hotel(self, url, f):
        '''open new page, gets description  and writes it'''
        chrome = self.load_driver(url) #new page
        '''description part'''
        hotel_name = chrome.find_element_by_class_name('hp__hotel-name').text  # finds class = 'hp__hotel-name' in HTML
        hotel_summary = chrome.find_element_by_id('summary').text  # finds id = 'summary' in HTML
        list_hotel_review = chrome.find_elements_by_class_name('hp-desc-review-highlight')  # not always
        hotel_review = ''

        for i in list_hotel_review:
            text = i.text
            hotel_review += text

        list_hotel_geo_info = chrome.find_elements_by_class_name('geo_information')  # not always
        hotel_geo_info = ''

        for i in list_hotel_geo_info:
            text = i.text
            hotel_geo_info += text

        chrome.close()
        '''writes'''
        f.write(hotel_name + '\n' + hotel_summary + '\n' + hotel_review + '\n' + hotel_geo_info + '\n' + 'testorcone'
                + '\n')

    def get_attributes(self, url, f):
        '''open new page, gets attributes  and writes them'''
        chrome = self.load_driver(url)  # new page
        '''attributes part'''
        hotel_name = chrome.find_element_by_class_name('hp__hotel-name').text
        hotel_attributes = ''
        list_hotel_attributes = chrome.find_elements_by_class_name('facilitiesChecklistSection')

        for i in list_hotel_attributes:
            text = i.text + '\nattributinuovi\n'
            hotel_attributes += text

        chrome.close()
        '''writes'''
        f.write(hotel_name + '\n' + hotel_attributes + '\n' + 'nuovohotelinarrivo' + '\n')

    def get_everything(self, url, f, g):
        #writes at the same 2 txt files: hotel descriptions and attributes"
        chrome = self.load_driver(url)
        '''attributes part'''
        hotel_attributes = ''
        list_hotel_attributes = chrome.find_elements_by_class_name('facilitiesChecklistSection')

        for i in list_hotel_attributes:
            text = i.text + '\nattributinuovi\n'
            hotel_attributes += text

        '''description part'''
        hotel_name = chrome.find_element_by_class_name('hp__hotel-name').text  # finds class = 'hp__hotel-name' in HTML
        hotel_summary = chrome.find_element_by_id('summary').text  # finds id = 'summary' in HTML
        list_hotel_review = chrome.find_elements_by_class_name('hp-desc-review-highlight')  # not always
        hotel_review = ''

        for i in list_hotel_review:
            text = i.text
            hotel_review += text

        list_hotel_geo_info = chrome.find_elements_by_class_name('geo_information')  # not always
        hotel_geo_info = ''

        for i in list_hotel_geo_info:
            text = i.text
            hotel_geo_info += text

        chrome.close()
        '''writes'''
        g.write(hotel_name + '\n' + hotel_attributes + '\n' + 'nuovohotelinarrivo' + '\n')
        f.write(hotel_name + '\n' + hotel_summary + '\n' + hotel_review + '\n' + hotel_geo_info + '\n' + 'testorcone'
                + '\n')

    def get_info_hotels(self, driver, f, g=0):
        '''from an initialized text file and driver wrties description/attributes or both'''
        last_page = int(driver.find_elements_by_class_name('sr_pagination_link')[-1].text)

        for j in range(1, last_page + 1):
            print('Currently in page ' + str(j))
            hotel_list = driver.find_elements_by_class_name('hotel_name_link')

            for i in hotel_list:
                url = i.get_attribute("href")

                if self.mode == 'attributes':
                    self.get_attributes(url, f)

                elif self.mode == 'description':
                    self.get_hotel(url, f)

                else:
                    self.get_everything(url, f, g)


            if j < last_page:
                next_page = driver.find_element_by_class_name('paging-next').get_attribute('href')
                driver.get(next_page)

        f.close()
        g.close()
        driver.close()

    def get_checkboxes(self, driver, f): #possible locations given a wide area(ex.Switzerland, Vaud)
        'not finished'
        checklist= ''
        filterbox = driver.find_elements_by_class_name('filterbox')

        for i in filterbox:
            text = i.text + '\nnuovocheckbox\n'
            checklist += text

        driver.close()
        f.write(checklist)

#test

booking = Crawler('./chromedriver', 'both')

myurl = 'https://www.booking.com/searchresults.en-gb.html?aid=356992&label=gog235jc-region_chalet-en-ch' \
        '-lakeNgenevaNregion-unspec-ch-com-L%3Aen-O%3AosSx-B%3Achrome-N%3AXX-S%3Abo-U%3Ac-H%3As&sid' \
        '=2c7b85390e1b1f40dd933b7e80c3c2de&sb=1&src=region&src_elem=sb&error_url' \
        '=https%3A%2F%2Fwww.booking.com%2Fregion%2Fch%2Flake-geneva-region.en-gb.html%3Faid%3D356992%3Blabel%3Dgog235jc' \
        '-region_chalet-en-ch-lakeNgenevaNregion-unspec-ch-com-L%253Aen-O%253AosSx-B%253Achrome' \
        '-N%253AXX-S%253Abo-U%253Ac-H%253As%3Bsid%3D2c7b85390e1b1f40dd933b7e80c3c2de%3Binac%3D0%3Bthm%3Dhotel%26%3B&region' \
        '=669&checkin_monthday=&checkin_month=&checkin_year=&checkout_monthday=&checkout_month=&checkout_year=&no_rooms=' \
        '1&group_adults=2&group_children=0&from_sf=1'
driver = booking.load_driver(myurl)
f = booking.write_file('nuovo.txt')
g = booking.write_file('attributes.txt')
booking.get_info_hotels(driver, f, g)
