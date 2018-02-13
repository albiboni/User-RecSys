import random
import numpy as np
#DATA:
#Note: after each list, it is indicated the amount of elements per list
#1.ATTRIBUTES
accomodation_type = ['holiday home', 'chalet', 'guest house', 'villa', 'bed and breakfast', 'hostel', 'apartment',
                     'hotel', 'pet-friendly accomodation'] #9

city = ['Bouveret', 'Les Rasses', 'Chesières', 'Rossinière', 'Sainte-Croix', 'Vevey', 'Leysin', 'Villeneuve',
        'Lausanne', 'Bellevue', 'Saint-Cergue', 'Gryon', 'Geneva', 'Nyon', 'Les Diablerets', 'Villars-sur-Ollon', 'Bex',
        'Morges', 'Yverdon-les-Bains', 'Montreux', 'Arveyes', 'Chateau-d’Oex', 'the airport'] #23
activity = ['play tennis', 'go skiing', 'play in casino', 'do horse riding', 'go cycling', 'go hiking',
            'play table tennis'] #7
outside_facility = ['sun terrace', 'balcony', 'terrace', 'patio', 'swimming pool', 'garden', 'beach nearby'] #7
view = ['with a view', 'with a view on the mountains', 'with a view on the lake'] #3
relax_facility = ['massage center', 'Hot Tub', 'Sauna', 'Hammam', 'swimming pool', 'fitness centre', 'library', 'Bath',
                  'sofa', 'fireplace', 'TV with cable channels', 'DVD player', 'CD player'] #13
hotel_facility = ['restaurant', 'laundry', 'ski school', 'lift', '24-hour reception', 'business centre', 'bar',
                  'Safety Deposit Box', 'shuttle service', 'room services', 'parking spot', 'children’s playground',
                  'breakfast'] #13
room_facility = ['sofa', 'coffee machine', 'kitchen', 'private bathroom', 'desk', 'washing machine', 'minibar',
                 'bathroom with a bath', 'refrigerator', 'microwave', 'toilet', 'hairdryer', 'air conditioning',
                 'WiFi', 'TV'] #15

#2.EXPRESSIONS
LOOK_FOR = ['I’m looking for', 'I would like', 'I would like to have', 'I would love to have',
            'I like to have', 'I seek', 'I request', 'I demand'] #8
CLOSE = ['in', ', which is in', 'close to', 'near', 'nearby', ', which is close to', ', which is nearby',
         ', which is near'] #8
CITY_ATTRIBUTE = ['the center of', 'the station of', 'the city center of', '', 'the city of', 'the heart of',
                  'the mall of', 'the town of', 'the marketplace of' ] #9
WANT_DO = ['I would like', 'I would love', 'I prefer', 'I’m addicted', 'I like', 'I love', 'I want', 'I desire'] #8
PERSONAL_PREFERENCE = ['to relax', 'to chill'] #2
LINKS_CONSEQUENCE = [', so', '. For this reason', ', in fact', ', indeed', 'therefore'] #5
LINKS_DIRECT = ['a', 'something like a', 'something such as a'] #3
LINKS_MORE = ['Similarly', 'In addition', 'Additionally', 'Furthermore', 'Besides', 'Moreover'] #6

APPRECIATE_IMPORTANCE= ['it would be appreciated if there was', 'it is important that there is', 'there must be',
                        'I appreciate', 'I care about', 'I request', 'I would like to have',
                        'I like to have', 'I would love to have', 'it would be nice to have'] #10
CONNECTORS = ['and', 'plus', 'as well as', 'and also', 'and including'] #5

#Writing functions
def write_expression(list1, f, g):
    """picks a random element from a list and writes it"""
    indeces = np.arange(np.shape(list1)[0])
    random_indeces = np.random.permutation(indeces)
    numpy_list = np.array(list1)
    shuffle_list = numpy_list[random_indeces]
    choice = np.random.randint(0,np.shape(list1)[0])
    attribute = shuffle_list[choice]
    f.write(attribute)
    if list1==accomodation_type:
        g.write(attribute)


def make_sentence(sequence, f, g):
    """write sentence iterating in the different lists"""
    for i in range(len(sequence)):
        f.write(' ')
        #g.write('\n')
        write_expression(sequence[i], f, g)

def write_attribute(attribute, f, all_features, g):
    """writes attributes and removes it from list"""
    f.write(' ')
    g.write('\n')
    f.write(attribute)
    g.write(attribute)
    all_features.remove(attribute)


#SENTENCES structure
sequence1 = [LOOK_FOR, LINKS_DIRECT,accomodation_type, CLOSE]
sequence2_view = [WANT_DO, ['to spend time outside'], LINKS_CONSEQUENCE, APPRECIATE_IMPORTANCE, LINKS_DIRECT,
                  outside_facility]
sequence2 = [WANT_DO, ['to spend time outside'], LINKS_CONSEQUENCE, APPRECIATE_IMPORTANCE, LINKS_DIRECT]
sequence3 = [LINKS_MORE, WANT_DO, ['to']]
sequence4 = [WANT_DO, PERSONAL_PREFERENCE, LINKS_CONSEQUENCE, APPRECIATE_IMPORTANCE, LINKS_DIRECT]
sequence_double = [CONNECTORS, LINKS_DIRECT]
sequence5 = [LOOK_FOR, ['a place with'], LINKS_DIRECT]
sequence6 = [LINKS_MORE, ['in my room'], APPRECIATE_IMPORTANCE, LINKS_DIRECT]

#inizialization of txt files
filename = 'user_queries.txt'
f = open(filename,'w')
filename1 = 'user_attributes.txt'
g = open(filename1, 'w')
#city_use=[]
for i in range(1000): #indicates number of user queries

    #Standard initial sentence regarding accomodation type and city
    city_use = city+[]

    random.shuffle(city_use)
    choose_city = random.choice(city_use)
    make_sentence(sequence1, f, g)
    g.write('\n')
    if choose_city!='the airport':
        make_sentence([CITY_ATTRIBUTE], f, g)
        write_attribute(choose_city, f, city_use, g)
    else:
        write_attribute(choose_city, f, city_use, g)
    f.write('.\n')
    g.write('\nnew_sentence\n')
    all_features = activity + outside_facility + view + relax_facility + hotel_facility + room_facility
    outside_facility_use = outside_facility + []
    #Random sentences generator from second sentence to a max of 5 sentences
    for i in range(np.random.randint(2,5)):
        'make random length, in this case it will add randomly 2 or 5 sentences'


        random.shuffle(all_features)
        attribute = random.choice(all_features)

        if attribute in outside_facility:
            make_sentence(sequence2, f, g)
            write_attribute(attribute, f, all_features, g)
            f.write('.\n')
            g.write('\nnew_sentence\n')

        elif attribute in view:
            make_sentence(sequence2, f, g)
            random.shuffle(outside_facility_use)
            choose_outside_facility = random.choice(outside_facility_use)
            write_attribute(choose_outside_facility, f, outside_facility_use, g)
            #make_sentence(sequence2_view, f, g)
            write_attribute(attribute, f, all_features, g)
            f.write('.\n')
            g.write('\nnew_sentence\n')

        elif attribute in activity:
            make_sentence(sequence3, f, g)
            write_attribute(attribute, f, all_features, g)
            f.write('.\n')
            g.write('\nnew_sentence\n')

        elif attribute in relax_facility:
            make_sentence(sequence4, f, g)
            write_attribute(attribute, f, all_features, g)
            f.write('.\n')
            g.write('\nnew_sentence\n')

        elif attribute in hotel_facility:
            make_sentence(sequence5, f, g)
            write_attribute(attribute, f, all_features, g)
            decision = random.choice(all_features)

            if decision in hotel_facility:
                make_sentence(sequence_double, f, g)
                write_attribute(decision, f, all_features, g)

            f.write('.\n')
            g.write('\nnew_sentence\n')

        elif attribute in room_facility:
            make_sentence(sequence6, f, g)
            write_attribute(attribute, f, all_features, g)
            decision = random.choice(all_features)

            if decision in room_facility:
                make_sentence(sequence_double, f, g)
                write_attribute(decision, f, all_features, g)

            f.write('.\n')
            g.write('\nnew_sentence\n')

    f.write('richiestautente\n')
    g.write('richiestautente\n')

f.close()
g.close()