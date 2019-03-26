#-*-coding:utf-8 -*- 
import requests
import re
import threading
import os
import json
from bs4 import BeautifulSoup
import hashlib
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

def compare_md5(filepath,md5s):
    if os.path.exists(filepath):
        md5file=open(filepath,'rb')
        md5=hashlib.md5(md5file.read()).hexdigest()
        md5file.close()
        if md5 in md5s:
            return True
        else:
            print('Warning:',name,'md5 do not match, we will try again')
            return False
    else:
        return False


def downloader(url,filenames,md5s,dir):
    for name in filenames:
        filepath  = os.path.join(dir,name)
        print('Download:',name)
        while not compare_md5(filepath,md5s):
            try:
                download(url+name,name,dir)
            except Exception as e:
                print('Warning:',name,'download failed! we will try again')
    
def rundownloader(url,filenames,md5s,dir,ThreadNum=5):
    perthread=int(len(filenames)/ThreadNum)
    for i in range(0,ThreadNum):
        t = threading.Thread(target=downloader,args=(url,filenames[perthread*i:perthread*(1+i)],md5s,dir,))
        t.start()
    t = threading.Thread(target=downloader,args=(url,filenames[perthread*ThreadNum:],md5s,dir,))
    t.start()


savedir = './sleep-edfx/sleep-telemetry'
url = 'https://physionet.org/physiobank/database/sleep-edfx/sleep-telemetry/'


md5s=open(os.path.join(savedir,'MD5SUMS.txt'),'rb')
md5s = md5s.read()
md5s=md5s.decode('utf-8')
md5s = md5s.split()

soup,page_info=RequestWeb(url)
links = soup.find_all('a',href=re.compile(r".edf"))
filenames = []
for link in links[1:]:
    begin = str(link).index('">')
    stop = str(link).index('</a>')
    filename = str(link)[begin+2:stop]
    filenames.append(filename)
rundownloader(url,filenames,md5s,savedir)

