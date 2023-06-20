import requests

r = requests.get(url='https://cs7ns1.scss.tcd.ie/', params={'shortname': 'dengji'})
print(r.text)