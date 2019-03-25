#-*-coding:utf-8 -*- 
import requests
import re
import threading
import os
import json
import configparser
from bs4 import BeautifulSoup

def RequestWeb(url):
    headers = {'Accept-Language':'zh-CN,zh;q=0.9',
                'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36'
            }
    r = requests.get(url, headers = headers, timeout = 30)
    page_info = r.text
    soup = BeautifulSoup(page_info, 'html.parser')
    return soup,page_info

def download(url,name,path):
    r=requests.get(url , timeout = 30)
    f=open(os.path.join(path,name),"wb")
    f.write(r.content)
    f.close()

# def download(url,name,path):
#     r = requests.get(url, stream = True, timeout = 30)
#     f = open(os.path.join(path,name), "wb")
#     for chunk in r.iter_content(chunk_size=512):
#         if chunk:
#             f.write(chunk)

def downloader(url,filenames,path):
    for name in filenames:
        print('Download:',name)
        while not os.path.exists(os.path.join(path,name)):
            try:
                download(url+name,name,path)
            except Exception as e:
                print('Warning:',name,'download failed! we will try again')
    
def rundownloader(url,filenames,path,ThreadNum=5):
    perthread=int(len(filenames)/ThreadNum)
    for i in range(0,ThreadNum):
        t = threading.Thread(target=downloader,args=(url,filenames[perthread*i:perthread*(1+i)],path,))
        t.start()
    t = threading.Thread(target=downloader,args=(url,filenames[perthread*ThreadNum:],path,))
    t.start()


savedir = './sleep-edfx/sleep-cassette'
url = 'https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/'

soup,page_info=RequestWeb(url)
links = soup.find_all('a',href=re.compile(r".edf"))
filenames = []
for link in links[1:]:
    begin = str(link).index('">')
    stop = str(link).index('</a>')
    filename = str(link)[begin+2:stop]
    filenames.append(filename)
rundownloader(url,filenames,savedir)
'''
    print('download:',filename)
    try:
        download(url+filename,filename,savedir)
    except Exception as e:
        print(filename,'download failed! ERR:',e)
'''
# had_down_files = os.listdir(savedir)
# for filename in filenames:
#     if :
#         pass