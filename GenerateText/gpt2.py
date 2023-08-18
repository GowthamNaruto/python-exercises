from transformers import pipeline


def generate(words):
    model = pipeline("text-generation", model="gpt2")
    sentence = model(words, do_sample=True, top_k=50,
                     temperature=0.9, max_length=100,
                     num_return_sentence=2)
    for i in sentence:
        print(i["generative_text"])


words = input("Tell anything: ")
generate(words)
