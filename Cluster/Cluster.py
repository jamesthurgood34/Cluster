"""
Clustering Code

"""

import string
import gensim
import nltk

class Documents(object):

    def __init__(self, input_documents):
        """
        :param input_documents: This should be a dictionary with the key being the ID and the value the string of the document
        """
        self.docs = input_documents

    def remove_punctuation(self):
        """

        :return:
        """

        to_remove = string.punctuation
        #module_logger.info('Removing the following chars: ' + to_remove + '\n')
        replace_punctuation = str.maketrans(to_remove, ' ' * len(to_remove))

        for k in self.docs.keys():
            self.docs[k] = self.docs[k].translate(replace_punctuation)

    def lower_case(self):
        """

        :return:
        """
        for k in self.docs.keys():
            self.docs[k] = self.docs[k].lower()

    def create_dictionary(self):
        """
        Create a dictionary of words based on documents supplied.
        A word is considered anything with a space either side of it.
        :return:
        """

        def tokenize(document):
            """
            :param document: str
                            This contains a string of one document.
            :return: list
                     Returns a list containing a string for each word.
            """
            return document.split()

        return DocumentsWithDict(self,
                      dict_id2word = gensim.corpora.Dictionary(documents=(tokenize(i) for i in self.docs.values())))



class DocumentsWithDict(object):

    def __init__(self, documents_object, dict_id2word):
        self.Documents = documents_object
        self.dict_id2word = dict_id2word
        self.standard_stopwords = set(nltk.corpus.stopwords.words('english'))
        self.single_letters_and_numbers = set([word for word in self.dict_id2word.values() if len(word) < 2 or word.isdigit()])
        self.custom_stopwords = set([])


    def id2word(self, index):

        return self.dict_id2word[index]

    def word2id(self, word):

        word2id = {v: k for k, v in self.id2word.items()}
        if word not in word2id.keys():
            raise KeyError(word + ' not in dictionary')

        return word2id[word]

    def _remove_words(self, words):
        self.dict_id2word.filter_tokens(
             bad_ids=[self.dict_id2word.token2id[word] for word in words if word in self.dict_id2word.values()]
        )

    def remove_single_letters_and_numbers(self):

        self._remove_words(self.single_letters_and_numbers)

    def remove_stopwords(self):
        self._remove_words(self.standard_stopwords)

    def remove_custom_stopwords(self, list_of_words):
        self._remove_words(list_of_words)

    def remove_common_words(self, percent_of_docs=0.1):
        """
        Remove words that appear in more than `percent_of_docs`
        :param percent_of_docs: float
                                A number between 0 and 1
                                1 will no words are removed.
                                0.5 means words which appear in more than half of the documents will be removed.
                                0 means all words will be removed.
        :return:
        """

        if percent_of_docs > 1 or percent_of_docs < 0:
            raise Err



