from better_profanity import profanity
text = input("Please enter the text: ")
censored = profanity.censor(text)
print(censored)
