# pip install countryinfo
from countryinfo import CountryInfo
import streamlit as st

st.set_page_config(
    page_title="Country Info",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/GowthamNaruto',
        'About': "# Country Information app"
    }
)
st.title("Country Info")
# Prompt the user for country
get_country = st.text_input("Enter country name")

# get_country = input("Enter your country: ")
country = CountryInfo(get_country)

# Initilize streamlit columns
# col1, col2, col3, col4, col5, = st.columns(5)

# Extract data from the library
capital = country.capital()
cur_arr = country.currencies()
currency = ', '.join(cur_arr)

lang_arr = country.languages()
languages = ', '.join(lang_arr)

bor_arr = country.borders()
borders = ', '.join(bor_arr)

other_arr = country.alt_spellings()
other_names = ', '.join(other_arr)

area = '{:,}'.format(country.area())

ph_arr = country.calling_codes()
ph_codes = ', '.join(ph_arr)

region = country.region()
subregion = country.subregion()

time_arr = country.timezones()
timezones = ', '.join(time_arr)

population = '{:,}'.format(country.population())
# Print the country informations
st.metric("Capital", capital)
st.metric("Region", region)
st.metric("Sub-Regione", subregion)
st.metric("Area in SKM", area)
st.metric("Population", population)
st.metric("Currency", currency)
st.metric("Language", languages)
st.metric("Timezones", timezones)
st.metric("Borders are", borders)
st.metric("Other names", other_names)


# Run "streamlit run info.py"
