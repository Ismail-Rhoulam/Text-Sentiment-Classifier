import math
import re




def clean(text):   # Removes HTML tags & punctuations from a given string
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)    
    # Remove punctuations
    text = re.sub(r'[^\w\s]', ' ', text)    
    return text


def processing(text):   # Processing and tokenizing a given string

    text = text.lower()
    text = clean(text)
    tokens = text.split()
    return tokens


def vocab_extract(text):    # Creating a list of vocabulary
    
    vocabulary = set()
    for words, label in text:
        vocabulary.update(words)
    return list(vocabulary)


def feat(text, vocabulary):     # Converting a list of word into a feature vector based on the provided vocabulary
    features = [0] * len(vocabulary)
    for word in text:
        if word in vocabulary:
            features[vocabulary.index(word)] += 1
    return features


def naive_bayes(features, labels, vocabulary):  # Trainning model
    samples = len(features)
    classes = len(set(labels))

    word_count = {c: [0] * len(vocabulary) for c in range(classes)}
    class_count = [0] * classes

    # Count word frequencies and class frequencies
    for f, label in zip(features, labels):
        class_count[label] += 1

        for i in range(len(vocabulary)):
            word_count[label][i] += f[i]

    # Calculate probabilities    
    word_probs = {c: [0] * len(vocabulary) for c in range(classes)}
    class_probs = [0] * classes

    for j in range(classes):
        for i in range(len(vocabulary)):
            
            # Avoiding 0 probabilities
            word_probs[j][i] = (word_count[j][i] + 1) / (class_count[j] + len(vocabulary))
        class_probs[j] = class_count[j] / samples

    return word_probs, class_probs


def prediction(features, word_probs, class_probs): # Predicting with the classifier
    predictions = []

    for feature in features:
        class_score = [0] * len(class_probs)

        for x in range(len(class_probs)):
            score = math.log(class_probs[x])    # Using log probabilities to avoid underflow

            for y in range(len(feature)):
                if feature[y] > 0:  # Only consider words that appear in the text
                    score += math.log(word_probs[x][y]) * feature[y]

            class_score[x] = score
        predictions.append(class_score.index(max(class_score)))

    return predictions