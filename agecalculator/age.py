import datetime


def ageCalculator(year, month, date):
    today = datetime.datetime.now().date()
    dob = datetime.date(year, month, date)
    age = int((today-dob).days / 365.25)
    print(f"Age: {age}")


year = int(input("Year: "))
month = int(input("Month: "))
date = int(input("Date: "))
ageCalculator(year, month, date)
