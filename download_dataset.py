#-*-coding:utf-8 -*- 
import requests
import re
import threading
import os
import hashlib
headers = {
            'User-Agent':'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/73.0.3683.75 Chrome/73.0.3683.75 Safari/537.36'
        }

def download(url,name,savedir):
    r=requests.get(url, headers, timeout = 30)
    f=open(os.path.join(savedir,name),"wb")
    f.write(r.content)
    f.close()

def compare_md5(filepath,md5s):
    if os.path.exists(filepath):
        try:
            md5file=open(filepath,'rb')
            md5=hashlib.md5(md5file.read()).hexdigest()
            md5file.close()
        except Exception as e:
            return False
        if md5 in md5s:
            return True
    else:
        return False


def downloader(url,filenames,md5s,dir):
    for name in filenames:
        filepath  = os.path.join(dir,name)
        print('Download:',name)
        while not compare_md5(filepath,md5s):
            try:
                download(url+name+'?download',name,dir)
            except Exception as e:
                print('Warning:',name,'download failed! we will try again')

def rundownloader(url,filenames,md5s,dir,ThreadNum=4):
    perthread=int(len(filenames)/ThreadNum)
    for i in range(0,ThreadNum):
        t = threading.Thread(target=downloader,args=(url,filenames[perthread*i:perthread*(1+i)],md5s,dir,))
        t.start()
    t = threading.Thread(target=downloader,args=(url,filenames[perthread*ThreadNum:],md5s,dir,))
    t.start()


savedir = './datasets/sleep-edfx/'
url = 'https://physionet.org/files/sleep-edfx/1.0.0/'
# https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf?download

MD5SUMS=open(os.path.join(savedir,'md5/sleep-cassette_MD5SUMS.txt'),'rb')
MD5SUMS = MD5SUMS.read()
MD5SUMS=MD5SUMS.decode('utf-8')
MD5SUMS = MD5SUMS.split()
md5s = MD5SUMS[::2]
filenames = MD5SUMS[::-2]
print('start download sleep-edfx/sleep-cassette')
rundownloader(url+'sleep-cassette/',filenames,md5s,savedir)

MD5SUMS=open(os.path.join(savedir,'md5/sleep-telemetry_MD5SUMS.txt'),'rb')
MD5SUMS = MD5SUMS.read()
MD5SUMS=MD5SUMS.decode('utf-8')
MD5SUMS = MD5SUMS.split()
md5s = MD5SUMS[::2]
filenames = MD5SUMS[::-2]
print('start download sleep-edfx/sleep-telemetry')
rundownloader(url+'sleep-telemetry/',filenames,md5s,savedir)

# soup,page_info=RequestWeb(url)
# links = soup.find_all('a',href=re.compile(r".edf"))
# filenames = []
# for link in links[1:]:
#     begin = str(link).index('">')
#     stop = str(link).index('</a>')
#     filename = str(link)[begin+2:stop]
#     filenames.append(filename)
#rundownloader(url,filenames,md5s,savedir)

