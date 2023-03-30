import utils,os,sys,numpy,json,cv2,lxml,bs4
from bs4 import BeautifulSoup
from utils import findweb
from utils import getart_name
print("please input the name of researcher")

name=input()

url=findweb(name)
art_name=getart_name(url)
print(art_name)





