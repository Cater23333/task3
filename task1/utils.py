def findweb(name):  #this function is to find the url of the author
    import requests,bs4
    from bs4 import BeautifulSoup

    name = name.replace(' ', '+')
    url="https://dblp.org/search?q="+name
    bs = BeautifulSoup(requests.get(url).content, "html.parser")
    for node in bs.find_all('ul', class_='result-list'):
         linklist = node.find_all('a')
         print(linklist[0]['href'])
         return linklist[0]['href']



def getart_name(url):  #this function is to get the title of article
    import requests, bs4
    from bs4 import BeautifulSoup
    #response = requests.get(url)
    bs = BeautifulSoup(requests.get(url).content, "html.parser")
    art_name=[]
    for node in bs.find_all('span', class_='title'):
        art_name.append(node.string)
    return art_name



