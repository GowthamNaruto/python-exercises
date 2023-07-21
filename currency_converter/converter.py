# pip install forex_python
from forex_python.converter import CurrencyRates

# Initiate the CurrecyRates function
conv = CurrencyRates()

# Prompt the user for the currencies  and amount that they have to convert
amount = int(input("Enter the amount: "))
from_currency = input("From Curency: ").upper()
to_currency = input("To Currency: ").upper()
print(from_currency, "To", to_currency, amount)

# Convert the given currencies
result = conv.convert(from_currency, to_currency, amount)

# Print the results
print(f"{to_currency}: {result}")
