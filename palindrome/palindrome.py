def plinndrome(sentence):
    for i in (",.'?/z><}{{}}'"):
        sentence = sentence.replace(i, "")
    palindrome = []
    words = sentence.split(' ')
    for word in words:
        word = word.lower()
        if word == word[::-1]:
            palindrome.append(word)
    return palindrome


sentence = input("Enter a sentence: ")
print(plinndrome(sentence))
