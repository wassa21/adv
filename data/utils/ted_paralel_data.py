from bs4 import BeautifulSoup, Comment
import json
import requests
import re

from time import sleep
from random import randint

import pandas as pd

from dateutil import parser
import sys

import pdb
from statistics import mode
# Set to "False" to avoid updating data. Set to "True" to update, which can be a lengthy process.
#Scraping specific data relies on general data, so it must be run at least once before specific is set to "True"
update_general_data = False
update_specific_data = True

if update_general_data == False:
    try:
        TED_gen_df = pd.read_csv('TEDGeneral.csv')
    except:
        print('No general data to read, set "fetch general data" to "True" to aquire data')

if update_specific_data == False:
    try:
        pd.read_csv('TEDSpecific.csv')
    except:
        print('No specific data to read, set "fetch specific data" to "True" to aquire data')


# Create function to show progress:

def show_progress(part, whole):
    """
    Input:
    part = element of list
    whole = list
    ---------
    Function:
    Find the number of element "part" is within "whole"
    ---------
    Output:
    Return the string "[nth list element] of [list length]"
    """
    path = len(whole)
    step = whole.index(part) + 1
    progress = str(step) + " of " + str(path)
    return progress



# Create URL binding functions
def get_num_pages(start_page_soup):
    """
    Input:
    start_page_soup = The HTML of a page 1 of talks on www.TED.com, as processed by BeautifulSoup
    ---------
    Function:
    Search the HTML of www.TED.com/talks for the number of pages of talks.
    ---------
    Output:
    The number of pages of talks on www.TED.com/talks
    """
    # There are hyperlinks to navigate pages of TED talks given the class "pagination", find that class
    pagination_str = start_page_soup.find(class_ = 'pagination')
    page_num_list = []
    # Step through the strings returned by searching for the pagination class and add them to a list
    for child in pagination_str.contents:
        page_num = re.search(r'\d+', str(child))
        if page_num != None:
            page_num_list.append(int(page_num[0]))
    # Return the value of link that contained the number of the largest page
    return max(page_num_list)

def build_page_urls(start_url, num_pages):
    """
    Input:
    start_url = The base URL that lists TED talks
    num_pages = the number of pages of TED talks
    ---------
    Function:
    Build a list of valid URLs of pages listing TED talks
    ---------
    Output:
    The list of URLs
    """
    url_list = []
    # Create strings of valid URLs according to the pattern observed on TED.com
    for i in range(1, num_pages + 1):
        url_list.append(re.sub(r'\d+', str(i), start_url))
    return url_list


# Functions to request webpage and handle errors
def request_webpage(url):
    """
    Input:
    url = The url to request
    ---------
    Function:
    Request the html from a URL while avoiding calling TED.com too frequently
    and while turning any errors that may occur over to an error handler.
    ---------
    Output:
    The html returned from a URL request
    """
    # If we receive a timeout, we're going to want to try again, timeout_tries is how many times
    timeout_tries = 5
    tries = 0
    # Count the number of tries, retry until we've tried enough
    while tries <= timeout_tries:
        #Wait a random interval to keep from calling the website too often.
        sleep(randint(10,25))
        #Try to get the page, if it returns a code that is not a success code, call the error handler
        try:
            page = requests.get(url, timeout=60)
            page_request_code = str(page.status_code)
            if page_request_code[:1] != '2':
                page = request_errorhandle(page_request_code)

            break
        # If the request times out, iterate up the number of tries and try again.
        except:
            tries = tries + 1

    return page

def request_errorhandle(code):
    """
    Input:
    code = The code of a failed URL request
    ---------
    Function:
    Handle the codes of any failed URL request that we are likely to receive
    ---------
    Output:
    The HTML returned from a URL request
    """
    # If we entered an invalid URL, return null data
    if code == '404':
        return ''
    # Most codes throw by valid URLs can be handled by waiting awhile and trying again
    else:
        sleep(randint(300,360))
        try:
            page = requests.get(url, timeout=60)
        #If that doesn't work, make the returned page data equal "error". I can then go back and fix it.
        except:
            page = 'error'

        return page



# Functions to retrieve data

def data_from_url(page_soup, data_dict):
    """
    Input:
    page_soup = The BeautifulSoup data from a page of TED talks
    data_dict = A dictionary of the data from TED talks that I will append to
    ---------
    Function:
    Extract the data I'm interested in and append it to a dictionary
    ---------
    Output:
    A dictionary containing the metadata I'm interested in from a page listing TED talks
    """
    # Save data from all talks on page to talk_data
    page_data = page_soup.find_all(class_ = 'talk-link')    
    # Iterate over that data and save the bits we're interested in
    for talk in page_data:
        try:
            media_data = talk.find('span' , {'class':'thumb thumb--video thumb--crop-top'})
            message_data = talk.find(class_ = 'media__message')
            title_url_data = message_data.find(class_ = ' ga-link', attrs = {'data-ga-context': 'talks'})
            meta_data = message_data.find('div', {'class':'meta'})
            
            #get title
            talk_title = title_url_data.contents[0].strip()
            #get speaker
            talk_speaker = message_data.find(class_ = 'h12 talk-link__speaker').contents[0].strip()  
            #get url
            talk_url = 'https://www.ted.com' + title_url_data['href'].strip()
            #get date posted
            talk_date = meta_data.find(class_ = 'meta__item').find(class_ = 'meta__val').contents[0].strip()
            #get rated bool and rating
            if meta_data.find(class_ = 'meta__row'):
                talk_rated_bool = True
                talk_ratings = meta_data.find(class_ = 'meta__row').find(class_ = 'meta__val').contents[0].strip()
            else:
                talk_rated_bool = False
                talk_ratings = None
            # get duration
            talk_duration = media_data.find('span', {'class':'thumb__duration'}).contents[0].strip()
            data_dict[talk_url] = ({'title': talk_title, 'speaker': talk_speaker, 'date': talk_date,
                                    'rated_bool': talk_rated_bool, 'ratings': talk_ratings,
                                    'duration': talk_duration})
        except:
            continue
    return data_dict

def extract_data_script(page_data, unique_start):
    """
    Input:
    page_data = The BeautifulSoup data from the page of a specific TED talk
    unique_start = Any unique string that marks the beginning of some javascript to extract
    ---------
    Function:
    Extract some javascript containing data I'm interested in and append it to a dictionary
    ---------
    Output:
    A string of raw javascript that contains data I'm interested in.
    (Further processing will be required to finalize data extraction.)
    """
    # Create a string from the BeautifulSoup page data.
    page_str = str(page_data)
    # Set the index to search within to the location of the string in unique_start
    start_idx = page_str.find(unique_start)
    # Find the closing and opening tags for all javascript
    num_script_start = re.finditer('<script', page_str)
    num_script_end = re.finditer('</script>', page_str)
    # Make a list of locations of opening and closing javacripts tags, then combine
    start_list = [s.start() for s in num_script_start]
    end_list = [e.end() for e in num_script_end]
    script_list = list(zip(start_list,end_list))

    # Narrow that list of locations down to just the ones that begin after unique_start
    narrowed_script_list = [s for s in script_list if s[0] >= start_idx ]

    # Ensure that we have selected the first opening tag and first closing tag after
    # unique start without getting mixed up somewhere along the way.
    if narrowed_script_list[0][1] < narrowed_script_list[1][0]:
        script_idx = narrowed_script_list[0]
    # If the closing tag cannot be found, just return the whole string
    else:
        narrowed_script_list = narrowed_script_list

    # Extract the javascript string immediately after unique_start
    data_extract = page_str[script_idx[0]:script_idx[1]]

    return data_extract



def get_transcript(transcript_page_soup, start_transcript, end_transcript):
    """
    Input:
    transcript_page_soup = The BeautifulSoup data from the page of a specific TED talk transcript
    start_transcript = Comment indicating the start of a transcript
    end_transcript = Comment indicating the end of a transcript
    ---------
    Function:
    Extract the transcript from a TED talk, if it exists
    ---------
    Output:
    A string containing the transcript of a TED talk.
    """
    # Initialize a list to store each element of a transcript
    transcript_list = []
    # In case there is no closing comment for a transcript, set a maximum
    # number of times we'll look for it.
    timeout = 300
    timeit = 0
    # Find all comments within the transcript page
    for comment in transcript_page_soup.find_all(text=lambda text:isinstance(text, Comment)\
                and text.strip() == start_transcript):
        # Find a comment that denotes the start of a transcript
        if str(comment).strip() == start_transcript:
            # Get the transcript until a comment is reached that denotes the end of a transcript
            while True:
                timeit = timeit + 1
                comment = comment.next_element
                if str(comment).strip() == end_transcript:
                    break
                transcript_list.append(comment)
                # If we've gone on for a while without finding a closing comment, end the loop
                if timeit >= timeout:
                    break
    # Join the list of elements that contained the transcript into a string
    transcript = ''.join(str(s) for s in transcript_list)
    return transcript


def get_transcript_separated(soup,start_transcript,end_transcript):
    data = []
    for comment in soup.find_all(string=lambda text:isinstance(text, Comment)):
        if comment.strip() == start_transcript:
            next_node = comment.next_sibling
            while next_node and next_node.next_sibling:
                val = str(next_node)
                s1 = val.find('<p>')
                s2 = val.find('</p>')
                if s1 != -1 and s2 != -1:
                    val = val[s1+3:s2]
                data.append(val)
                #data.append(next_node)
                next_node = next_node.next_sibling
                if not next_node.name and next_node.strip() == end_transcript: break;
    return data

def get_translation_langues(soup):
    langs = {}
    for i in soup.findAll('link'):
        if i.get('href')!=None and i.attrs['href'].find('?language=')!=-1:
            lang=i.attrs['hreflang']
            path=i.attrs['href']
            langs[lang] = path
    return langs

# Check to see if I've toggled update_general_data on.
if update_general_data == True:
    # Retrieve the first page of TED talks
    start_url = 'https://www.ted.com/talks?page=1'
    start_page = request_webpage(start_url)
    start_page_soup = BeautifulSoup(start_page.text, 'html.parser')
    # Use that page to find the total number of pages
    num_talk_pages = get_num_pages(start_page_soup)    
    # Build a list of URLs for all the pages listing TED talks
    talks_url_list = build_page_urls(start_url, num_talk_pages)



# Check to see if I've toggled update_general_data on.
if update_general_data == True:
    # Set up dictionary for temporary storage
    TED_talk_dict = {}
    for url in talks_url_list:
        try:
            # Call webpage request function
            page = request_webpage(url)
            # Convert page request to BeautfulSoup
            page_soup = BeautifulSoup(page.text, 'html.parser')      
            # Pull data from soup using data_url_fuction and add to dict
            TED_talk_dict = data_from_url(page_soup, TED_talk_dict)
            # Show progress
            sys.stdout.write('\r'+ 'step ' + show_progress(url, talks_url_list))
        except:
            print("Problem appeared while creating urls:{}".format(url))

            
    # Make dataframe of all data from URLs
    TED_gen_df = pd.DataFrame.from_dict(TED_talk_dict, orient='index').reset_index()
    # Process dataframe, setting the URL (which must be unique) to the index
    TED_gen_df['url'] = TED_gen_df['index']
    TED_gen_df = TED_gen_df.drop('index', axis=1)

    # Show a sample
    TED_gen_df.sample(15)

# Check to see if I've toggled update_general_data on.
if update_general_data == True:
    #If so, save our newly retrieve general data to a CSV
    TED_gen_df.to_csv('TEDGeneral.csv', index=False)


target_langs = ['en','tr','fr','de','pt','es']
# Add two empty columns to our dataframe
TED_gen_df['raw_data'] = ''
TED_gen_df['transcript'] = ''
# For each TED talk in the dataframe, retrieve its URL (stored as index)
errors = []
for idx in TED_gen_df.index:
    try:
        page_url      = TED_gen_df.iloc[idx]['url']
        page_title    = TED_gen_df.iloc[idx]['title']
        page_speaker  = TED_gen_df.iloc[idx]['speaker']
        page_duration = TED_gen_df.iloc[idx]['duration']
        # Make the url for the transcript page
        transcript_url = page_url + '/transcript'
        # Request web pages for the specific talk data and the transcript data.
        talk_page = request_webpage(page_url)
        talk_page_soup = BeautifulSoup(talk_page.text,'html.parser')
        page_tags = ",".join([x["content"] for x in talk_page_soup.find_all('meta',property="og:video:tag")])
        extracted_data = ''
        langs = get_translation_langues(talk_page_soup)
        paralel_data = {}
        for l in langs:
            if l in target_langs:
                lang_url = langs[l]
                lang_url = lang_url.replace('?language=','/transcript?language=')
                transcript_page = request_webpage(lang_url)
                # If the transcript page is not blank, and an error was not returned, extract transcript from it
                if transcript_page != '' and transcript_page != 'error':
                    transcript_page_soup = BeautifulSoup(transcript_page.text, 'html.parser')
                    extracted_transcript = get_transcript_separated(transcript_page_soup, 'Transcript text', '/Transcript text')
                    paralel_data[l] = extracted_transcript

        # Most frequent length'i bul
        # o lengthe sahip olan elemanlari paralel_data_new'e al.
        _abc = [len(paralel_data[x]) for x in paralel_data.keys()]
        most_freq_len = mode(_abc)
        paralel_data_new = {}
        for l in paralel_data.keys():
            if len(paralel_data[l]) == most_freq_len:
                paralel_data_new[l] = paralel_data[l]
        
        df_talk = pd.DataFrame.from_dict(paralel_data_new)
        df_talk['title']   =  page_title
        df_talk['speaker'] =  page_speaker
        df_talk['duration'] = page_duration
        df_talk['tags']     = page_tags
        csv_name = './tedtalks/paralel_talks_ted_{}_ix.csv'.format(idx)
        df_talk.to_csv(csv_name,index=False,sep=',')
        #Show progress. Blank string is to ensure complete overwrite of previous string after carriage return.
        sys.stdout.write('\r' + 'step ' + show_progress(idx, list(TED_gen_df.index)) + ': ' + transcript_url\
                         +'                                                                                 ')
    except:
        print("error on talk with index:{}".format(idx))
        errors.append(idx)

#print(",".join(errors))





