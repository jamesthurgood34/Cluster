"""
Clustering Code

"""

import string
import gensim

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


        DocumentsWithDict(self,
                          dict_id2word = gensim.corpora.Dictionary(documents=(tokenize(i) for i in self.docs.values)))



class DocumentsWithDict(object):

    def __init__ (self, documents_object, dict_id2word):
        self.Documents = documents_object
        self.dict_id2word = dict_id2word

    def id2word(self, id):
        return self.dict_id2word[id]

    def word2id(self, word):
        word2id = {v : k for k, v in self.id2word.items()}

        if word not in word2id.keys():
            raise KeyError(word + ' not in dictionary')

        return word2id[word]



