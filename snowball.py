import nltk
import random

from collections import Counter
from nltk.corpus import brown
from nltk.tag import pos_tag, map_tag
from nltk.tokenize import word_tokenize

# #['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 
# # 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction']

# fiction = brown.words(categories="fiction")
# romance = brown.words(categories="romance")

def read_file(filename):
    with open(filename, "r") as fp:
        contents = fp.readlines()
        contents = list(map(lambda s: s.strip("\n"), contents))
        contents = list(filter(None, contents))
        joint_words = ' '.join(contents)
        separated_words = word_tokenize(joint_words)
        return separated_words

def clean_text(text):
    punc = "!#$%&()*+,-./:;<=>?@[\]^_`{|}~"
    text = " ".join(ch.lower() for ch in text if ch not in punc)
    text = text.split()
    return text

def create_bigrams(text):
    bigrams = {}

    current_word = "$"

    for next_word in text:

        if current_word not in bigrams:
            bigrams[current_word] = [next_word]
        else:
            bigrams[current_word] += [next_word]

        current_word = next_word

    return bigrams

def create_tagrams(tags):
    tagsets = {}

    current_tag = "$"

    for next_tag in tags:
        if current_tag not in tagsets:
            tagsets[current_tag] = [next_tag]
        else:
            tagsets[current_tag] += [next_tag]

        current_tag = next_tag

    return tagsets

def count_tag_frequency(tagsets):
    frequencies = {}

    for key, val in tagsets.items():
        frequencies[key] = Counter(val)

    return frequencies

def generate_bigram_pairs(bigrams):

    pairs = []

    for key, val in bigrams.items():
        for word in val:
            if len(word) - len(key) == 1:
                pairs.append([key, word])

    pairs.sort(key=lambda x: len(x[0]))

    generations = {}

    for pair in pairs:
        lengths = str(len(pair[0])) + ":" + str(len(pair[1]))

        if lengths not in generations:
            generations[lengths] = [pair]
        else:
            generations[lengths].append(pair)

    return generations

def generate_next_word(first, second, frequencies):
    first_tag = nltk.tag.pos_tag(first)
    second_tag = nltk.tag.pos_tag(second)

    for key, val in frequencies[first_tag[0][-1]].items():
        if key == first_tag[1][-1]:
            f_t = val
        if key == second_tag[0][-1]:
            s_t = val

    if f_t > s_t:
        return [first[0], first[1]]
    else:
        return [first[0], second[0]]

def chain(prior, generations_val_1, generations_val_2, frequencies):
    all_ = []
    for gen in generations_val_1:
        if prior[-1] in gen[0]:
            all_.append(gen)
    pick = random.choice(all_)
    next = random.choice(generations_val_2)
    return generate_next_word(pick, next, frequencies)

if __name__ == "__main__":


    text = brown.words(categories="news")
    # romance = brown.words(categories="romance")
    # learned = brown.words(categories="learned")
    # news = brown.words(categories="mystery")

    cleaned_text = clean_text(text)
    
    bigrams = create_bigrams(cleaned_text)
    
    tagged_text = nltk.tag.pos_tag(text)
    tags = [i[1] for i in tagged_text]

    tagsets = create_tagrams(tags)
    
    frequencies = count_tag_frequency(tagsets)
    
    generations = generate_bigram_pairs(bigrams)


    first = random.choice(generations["1:2"])
    second = random.choice(generations["2:3"])

    start = generate_next_word(first, second, frequencies)

    c1 = chain(start, generations["2:3"], generations["3:4"], frequencies)
    c2 = chain(c1, generations["3:4"], generations["4:5"], frequencies)
    c3 = chain(c2, generations["4:5"], generations["5:6"], frequencies)
    c4 = chain(c3, generations["5:6"], generations["6:7"], frequencies)
    c5 = chain(c4, generations["6:7"], generations["7:8"], frequencies)
    c6 = chain(c5, generations["7:8"], generations["8:9"], frequencies)

    print(start[0], start[-1], c1[-1], c2[-1], c3[-1], c4[-1], c5[-1], c6[-1])


