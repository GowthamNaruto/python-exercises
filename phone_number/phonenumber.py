import phonenumbers as ph
from phonenumbers import carrier
from phonenumbers import geocoder
from phonenumbers import timezone

number = "+919443560415"
number = ph.parse(number)
print(f"Timezone: {timezone.time_zones_for_number(number)}")
print(f"Carrier: {carrier.name_for_number(number, 'en')}")
print(f"Country: {geocoder.description_for_number(number, 'en')}")
