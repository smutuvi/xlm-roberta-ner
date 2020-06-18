from core.polrobertner import PolRobertaNer


def main():
    ner = PolRobertaNer("models/kpwr_n82_base", "models/roberta_base_fairseq")

    sentences = ["Ala z Krakowa jeździ Audi",
                 "Marek Nowak z Politechniki Wrocławskiej mieszka przy ul . Sądeckiej"]
    sentences_tokens = [sentence.split(" ") for sentence in sentences]
    sentences_labels = ner.process(sentences_tokens)

    print("-" * 20)
    for sentence_tokens, sentence_labels in zip(sentences_tokens, sentences_labels):
        for token, label in zip(sentence_tokens, sentence_labels):
            print("%-12s %s" % (token, label))
        print("-"*20)


if __name__ == "__main__":
    try:
        main()
    except ValueError as er:
        print("[ERROR] %s" % er)
