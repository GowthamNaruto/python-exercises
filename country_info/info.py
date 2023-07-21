# pip install countryinfo
from countryinfo import CountryInfo

# Prompt the user for country
get_country = input("Enter your country: ")
country = CountryInfo(get_country)

# Print the country informations
print(f"Capital: {country.capital()}")
print(f"Currency: {country.currencies()}")
print(f"Language: {country.languages()}")
print(f"Borders are: {country.borders()}")
print(f"Other names: {country.alt_spellings()}")
