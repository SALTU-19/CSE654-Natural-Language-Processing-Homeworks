import re
import numpy as np

import math


def generate_ngrams(s, n):
    # Convert to lowercases
    s = s.lower()

    # Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)

    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != ""]

    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


consonant = ["b", "c", "d", "g", "ğ", "j", "l", "m", "n", "r",
             "v", "y", "z", "ç", "f", "h", "k", "p", "s", "ş", "t"]
vowel = ["a", "ı", "o", "u", "e", "i", "ö", "ü"]


def parser(word):
    spell = list(word)
    vowel_count = sum([1 for i in spell if i in vowel])
    lenght = len(word)

    if lenght == 1 or lenght == 2:
        return word

    if (lenght == 4 or lenght == 3) and vowel_count == 1:
        return word

    if word[0] in consonant:
        if word[1] in consonant:
            if word[4] in consonant:
                return word[:4] + " " + parser(word[4:])
            else:
                return word[:3] + " " + parser(word[3:])
        else:
            if word[2] in consonant and word[3] in consonant and not word[4] in consonant:
                return word[:3] + " " + parser(word[3:])
            elif word[2] in consonant and word[3] in consonant and word[4] in consonant:
                return word[:4] + " " + parser(word[4:])
            else:
                return word[:2] + " " + parser(word[2:])
    else:
        if word[1] in vowel:
            return word[0] + " " + parser(word[1:])
        else:
            if word[2] in vowel:
                return word[:1] + " " + parser(word[1:])
            else:
                return word[:2] + " " + parser(word[2:])

# parse string into syllables


def parse_syllable(string):
    # read the file split the words
    words = []
    for line in string.splitlines():
        for word in line.split(" "):
            word = turkish_to_english(word)
            if(len(word) >= 1):
                words.append(word)

    # collect parsed words
    parsed_words = []
    for word in words:
        parsed_words.append(parser(word))

    # append parsed words to a single string
    parsed_words_s = " ".join(parsed_words)
    return parsed_words_s

# read all file to a single string


def read_file(filename):
    with open(filename, mode="r", encoding="utf-8") as f:
        corpus = f.read()
    return corpus

# convert turkish letters to english letters


def turkish_to_english(string):
    choices = {"İ": "I", "ş": "s", "Ş": "S", "ı": "i", "ö": "o", "ü": "u", "ç": "c", "ğ": "g", "Ç": "C", "Ö": "O", "Ü": "U", "Ğ": "G", "â": "a", "î": "i", "û": "u", "ê": "e", "ô": "o", "Â": "A", "Î": "I", "Û": "U", "Ê": "E", "Ô": "O", "â": "a", "î": "i", "û": "u", "ê": "e", "ô": "o", "Â": "A", "Î": "I", "Û": "U", "Ê": "E", "Ô": "O", "â": "a", "î": "i", "û": "u",
               "ê": "e", "ô": "o", "Â": "A", "Î": "I", "Û": "U", "Ê": "E", "Ô": "O", "â": "a", "î": "i", "û": "u", "ê": "e", "ô": "o", "Â": "A", "Î": "I", "Û": "U", "Ê": "E", "Ô": "O", "â": "a", "î": "i", "û": "u", "ê": "e", "ô": "o", "Â": "A", "Î": "I", "Û": "U", "Ê": "E", "Ô": "O", "â": "a", "î": "i", "û": "u", "ê": "e", "ô": "o", "Â": "A", "Î": "I", "Û": "U"}
    for i in range(len(string)):
        string = string.replace(
            string[i:i+1], choices.get(string[i], string[i]))
    return string


def parse_string_two(string):
    return string.split(" ")[1]
# parse string to bigrams


def parse_string_three(string):
    return string.split(" ")[2]


def count_element_matrix(matrix, element):
    count = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if(matrix[i][j] == element):
                count += 1
    return count


def good_turing_smooting(ngram_matrix, ngrams, unique_ngrams):
    gt_smooth = np.zeros((len(unique_ngrams), len(unique_ngrams)))
    # calculate good turing smoothing
    for i in range(len(ngram_matrix)):
        for j in range(len(ngram_matrix[i])):
            if(ngram_matrix[i][j] == 0):
                gt_smooth[i][j] = count_element_matrix(
                    ngram_matrix, 1) / len(ngrams)
            else:
                gt_smooth[i][j] = (ngram_matrix[i][j]+1) * \
                    count_element_matrix(ngram_matrix, ngram_matrix[i][j]+1) / \
                    count_element_matrix(ngram_matrix, ngram_matrix[i][j])
    return gt_smooth


def generate_bigram_matrix(unique_bigrams, bigrams):
    bigram_matrix = np.zeros((len(unique_bigrams), len(unique_bigrams)))
    for i in range(len(bigrams)-1):
        bigram_matrix[unique_bigrams.index(
            bigrams[i])][unique_bigrams.index(bigrams[i+1])] += 1
    gt_smooth = good_turing_smooting(bigram_matrix, bigrams, unique_bigrams)

    # edit bigram_matrix to make it more accurate
    # for i in range(len(gt_smooth)):
    #     if(np.sum(gt_smooth[i]) != 0):
    #         gt_smooth[i] = gt_smooth[i] / np.sum(gt_smooth[i])
    return gt_smooth


def generate_towgram_matrix(unique_towgrams, towgrams, unique_bigrams):
    towgram_matrix = np.zeros((len(unique_towgrams), len(unique_bigrams)))
    for i in range(len(towgrams)-1):
        towgram_matrix[unique_towgrams.index(
            towgrams[i])][unique_bigrams.index(parse_string_two(towgrams[i+1]))] += 1

    gt_smooth = good_turing_smooting(towgram_matrix, towgrams, unique_towgrams)
    # edit towgram_matrix to make it more accurate
    # for i in range(len(towgram_matrix)):
    #     if(np.sum(towgram_matrix[i]) != 0):
    #         towgram_matrix[i] = towgram_matrix[i] / np.sum(towgram_matrix[i])
    return gt_smooth


def generate_threegram_matrix(unique_threegrams, threegrams, unique_bigrams):
    threegram_matrix = np.zeros((len(unique_threegrams), len(unique_bigrams)))
    for i in range(len(threegrams)-1):
        threegram_matrix[unique_threegrams.index(
            threegrams[i])][unique_bigrams.index(parse_string_three(threegrams[i+1]))] += 1

    gt_smoothing = good_turing_smooting(
        threegram_matrix, threegrams, unique_threegrams)
    # edit threegram_matrix to make it more accurate
    # for i in range(len(threegram_matrix)):
    #     if(np.sum(threegram_matrix[i]) != 0):
    #         threegram_matrix[i] = threegram_matrix[i] / \
    #             np.sum(threegram_matrix[i])
    return gt_smoothing


def total_pair(ngrams):
    total = 0
    for i in range(len(ngrams)-1):
        total += 1
    return total


def probab(string, bigrams, unique_bigrams):
    count = 0
    for i in range(len(bigrams)-1):
        if bigrams[i] == string:
            count += 1
    return count/len(unique_bigrams)


def chain_rule_bigram(bigram_matrix, search, bigrams, unique_bigrams, index, prob):
    if index == len(search)-1:
        return prob * probab(search[0], bigrams, unique_bigrams)
    else:
        prob = prob * \
            bigram_matrix[unique_bigrams.index(
                search[index])][unique_bigrams.index(search[index+1])]
        return chain_rule_bigram(bigram_matrix, search, bigrams, unique_bigrams, index+1, prob)


def perplexity_bigram(bigram_matrix, search, bigrams, unique_bigrams, index, perp):
    if index == len(search)-1:
        perp = perp + math.log2(probab(search[0], bigrams, unique_bigrams))
        return pow(2, -perp)
    else:
        perp = perp + \
            math.log2(bigram_matrix[unique_bigrams.index(
                search[index])][unique_bigrams.index(search[index+1])])
        return perplexity_bigram(bigram_matrix, search, bigrams, unique_bigrams, index+1, perp)


def chain_rule_towgram(towgram_matrix, bigram_matrix, search, towgrams, bigrams, unique_towgrams, unique_bigrams, index, prob):
    if index == len(search)-1:
        return prob * probab(search[0].split(" ")[0], bigrams, unique_bigrams) * bigram_matrix[unique_bigrams.index(
            search[0].split(" ")[0])][unique_bigrams.index(search[0].split(" ")[1])]
    else:
        prob = prob * \
            towgram_matrix[unique_towgrams.index(
                search[index])][unique_bigrams.index(parse_string_two(search[index+1]))]
        return chain_rule_towgram(towgram_matrix, bigram_matrix, search, towgrams, bigrams, unique_towgrams, unique_bigrams, index+1, prob)


def perplexity_towgram(towgram_matrix, bigram_matrix, search, towgrams, bigrams, unique_towgrams, unique_bigrams, index, perp):
    if index == len(search)-1:
        perp = perp + math.log2(probab(search[0].split(" ")[0], bigrams, unique_bigrams)) + math.log2(bigram_matrix[unique_bigrams.index(
            search[0].split(" ")[0])][unique_bigrams.index(search[0].split(" ")[1])])
        return pow(2, -perp/2)
    else:
        perp = perp + \
            math.log2(towgram_matrix[unique_towgrams.index(
                search[index])][unique_bigrams.index(parse_string_two(search[index+1]))])
        return perplexity_towgram(towgram_matrix, bigram_matrix, search, towgrams, bigrams, unique_towgrams, unique_bigrams, index+1, perp)


def chain_rule_threegram(threegram_matrix, towgram_matrix, bigram_matrix, search, threegrams, towgrams, bigrams, unique_threegrams, unique_towgrams, unique_bigrams, index, prob):
    if index == len(search)-1:
        return prob * probab(search[0].split(" ")[0], bigrams, unique_bigrams) * bigram_matrix[unique_bigrams.index(
            search[0].split(" ")[0])][unique_bigrams.index(search[0].split(" ")[1])] * towgram_matrix[unique_towgrams.index(
                search[0].split(" ")[0]+" "+search[0].split(" ")[1])][unique_bigrams.index(search[0].split(" ")[2])]
        # return prob
    else:
        prob = prob * \
            threegram_matrix[unique_threegrams.index(
                search[index])][unique_bigrams.index(parse_string_three(search[index+1]))]
        return chain_rule_threegram(threegram_matrix, towgram_matrix, bigram_matrix, search, threegrams, towgrams, bigrams, unique_threegrams, unique_towgrams, unique_bigrams, index+1, prob)


def perplexity_threegram(threegram_matrix, towgram_matrix, bigram_matrix, search, threegrams, towgrams, bigrams, unique_threegrams, unique_towgrams, unique_bigrams, index, perp):
    if index == len(search)-1:
        perp = perp + math.log2(probab(search[0].split(" ")[0], bigrams, unique_bigrams)) + math.log2(bigram_matrix[unique_bigrams.index(
            search[0].split(" ")[0])][unique_bigrams.index(search[0].split(" ")[1])]) + math.log2(towgram_matrix[unique_towgrams.index(
                search[0].split(" ")[0]+" "+search[0].split(" ")[1])][unique_bigrams.index(search[0].split(" ")[2])])
        return pow(2, -perp/3)
    else:
        perp = perp + \
            math.log2(threegram_matrix[unique_threegrams.index(
                search[index])][unique_bigrams.index(parse_string_three(search[index+1]))])
        return perplexity_threegram(threegram_matrix, towgram_matrix, bigram_matrix, search, threegrams, towgrams, bigrams, unique_threegrams, unique_towgrams, unique_bigrams, index+1, perp)


def main():
    # read file
    fileString = read_file("words.txt")
    # parse string into syllables
    parsed_words_file = parse_syllable(fileString)
    # collect bigrams
    n = 1
    bigrams = generate_ngrams(parsed_words_file, n)
    # collect two grams
    n = 2
    towgrams = generate_ngrams(parsed_words_file, n)
    # collect three grams
    n = 3
    threegrams = generate_ngrams(parsed_words_file, n)

    # collect unique bigrams in a list
    unique_bigrams = []
    for grams in bigrams:
        if grams not in unique_bigrams:
            unique_bigrams.append(grams)
    # collect unique two grams in a list
    unique_towgrams = []
    for grams in towgrams:
        if grams not in unique_towgrams:
            unique_towgrams.append(grams)
    # collect unique three grams in a list
    unique_threegrams = []
    for grams in threegrams:
        if grams not in unique_threegrams:
            unique_threegrams.append(grams)

    print("bigrams: ")
    print(bigrams)
    print("unique_bigrams: ")
    print(unique_bigrams)
    print("bigram_matrix: ")
    bigram_matrix = generate_bigram_matrix(unique_bigrams, bigrams)
    print(bigram_matrix)
    print("----------------------------------")
    print("towgrams: ")
    print(towgrams)
    print("unique_towgrams: ")
    print(unique_towgrams)
    print("towgram_matrix: ")
    towgram_matrix = generate_towgram_matrix(
        unique_towgrams, towgrams, unique_bigrams)
    print(towgram_matrix)
    print("----------------------------------")
    print("threegrams: ")
    print(threegrams)
    print("unique_threegrams: ")
    print(unique_threegrams)
    print("threegram_matrix: ")
    threegram_matrix = generate_threegram_matrix(
        unique_threegrams, threegrams, unique_bigrams)
    print(threegram_matrix)
    print("----------------------------------")

    string = "sakat salata"
    parsed_string = parse_syllable(string)
    print("parsed_string: ")
    print(parsed_string)
    string_bigrams = generate_ngrams(parsed_string, 1)
    print("string_bigrams: ")
    print(string_bigrams)
    prob_bigram = chain_rule_bigram(
        bigram_matrix, string_bigrams, bigrams, unique_bigrams, 0, 1)
    print("prob_bigram: ")
    print(prob_bigram)
    perp_bigram = perplexity_bigram(
        bigram_matrix, string_bigrams, bigrams, unique_bigrams, 0, 0)
    print("perp_bigram: ")
    print(perp_bigram)
    print("----------------------------------")
    print("string_towgrams: ")
    string_towgrams = generate_ngrams(parsed_string, 2)
    print(string_towgrams)
    prob_twogram = chain_rule_towgram(
        towgram_matrix, bigram_matrix, string_towgrams, towgrams, bigrams, unique_towgrams, unique_bigrams, 0, 1)
    print("prob_twogram: ")
    print(prob_twogram)
    perp_twogram = perplexity_towgram(
        towgram_matrix, bigram_matrix, string_towgrams, towgrams, bigrams, unique_towgrams, unique_bigrams, 0, 0)
    print("perp_twogram: ")
    print(perp_twogram)
    print("----------------------------------")
    print("string_threegrams: ")
    string_threegrams = generate_ngrams(parsed_string, 3)
    print(string_threegrams)
    prob_threegram = chain_rule_threegram(threegram_matrix, towgram_matrix, bigram_matrix, string_threegrams,
                                          threegrams, towgrams, bigrams, unique_threegrams, unique_towgrams, unique_bigrams, 0, 1)
    print("prob_threegram: ")
    print(prob_threegram)
    perp_threegram = perplexity_threegram(threegram_matrix, towgram_matrix, bigram_matrix, string_threegrams,
                                          threegrams, towgrams, bigrams, unique_threegrams, unique_towgrams, unique_bigrams, 0, 0)
    print("perp_threegram: ")
    print(perp_threegram)
    print("----------------------------------")


main()