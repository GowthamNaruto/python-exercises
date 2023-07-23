# pip install bs4
import requests
from bs4 import BeautifulSoup

# Prompt for the url
url = input("Enter the url: ")
r = requests.get(url)

# Parse the url
soup = BeautifulSoup(r.content, 'html.parser')
# Find all the a-tags
links = soup.find_all('a')

print(f"Here are the links from the '{url}' url")

# Print the links
for link in links:
    print(link.get('href'))
