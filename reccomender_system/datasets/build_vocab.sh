#!/bin/bash
#alphabetical order with number of appereance and word
# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names

cat ./pp/hotel_descriptions.txt ./pp/user_queries.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > ./fasttext/vocab.txt
