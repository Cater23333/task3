def findweb(name):  #this function is to find the url of the author
    import requests,bs4
    from bs4 import BeautifulSoup
    url="https://dblp.org/"
    response = requests.get(url)

def getart_name(url):  #this function is to get the title of article
    import requests, bs4
    from bs4 import BeautifulSoup
    #response = requests.get(url)
    bs = BeautifulSoup(requests.get(url).content, "html.parser")
    art_name=[]
    for node in bs.find_all('span', class_='title'):
        print(art_name.append(node.string))
        print('\n')
    return art_name



