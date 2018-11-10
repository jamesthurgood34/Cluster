"""
Clustering Code

"""

import string
import gensim
import nltk
import logging
import pandas as pd

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class Documents(object):

    def __init__(self, input_documents):
        """
        :param input_documents: This should be a dictionary with the key being the ID and the value the string of the
        document
        """
        self.docs = input_documents
        logger.debug('Created documents instance')


    def remove_punctuation(self):
        """

        :return:
        """

        to_remove = string.punctuation
        logger.info('Removing the following chars: ' + to_remove + '\n')
        replace_punctuation = str.maketrans(to_remove, ' ' * len(to_remove))

        for k in self.docs.keys():
            self.docs[k] = self.docs[k].translate(replace_punctuation)

        return self

    def lower_case(self):
        """

        :return:
        """
        for k in self.docs.keys():
            self.docs[k] = self.docs[k].lower()

        return self

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
        """
        This inherits the documents object and then precomputes variables used later in the class.


        :param documents_object: Documents object.
        :param dict_id2word: Dictionary created from documents object
        """

        self.Documents = documents_object
        self.dict_id2word = dict_id2word
        self.standard_stopwords = set(nltk.corpus.stopwords.words('english'))
        self.single_letters_and_numbers = set([word for word in self.dict_id2word.values() if len(word) < 2 or word.isdigit()])
        self.custom_stopwords = set([])
        self.removed_words = []



    def id2word(self, index):
        """
        This returns the word that corresponds to a given index.
        :param index: int
                        Index that corresponds to a word in the dictionary.
        :return: str

        """

        return self.dict_id2word[index]

    def word2id(self, word):
        """
        THis returns the index that corresponds to a given word.
        :param word: str
                        A word which you would like to find the index for.
        :return: int
        """

        if word not in self.dict_id2word.token2id.keys():
            raise KeyError(word + ' not in dictionary')

        return self.dict_id2word.token2id[word]

    def _remove_words_or_IDs(self, words_or_IDs):
        """
        Utility function to remove a list of words from the dictionary.
        :param words: [str/int]
                    A list of words or IDs which you would like to remove from the dictionary.
        :return:
        """

        #is the list a list of ID's?
        if all([isinstance(i, int) for i in words_or_IDs]):
            bad_ids = words_or_IDs
        else:
            words = words_or_IDs
            bad_ids = [self.dict_id2word.token2id[word] for word in words if word in self.dict_id2word.values()]

        #Keep this information for later, once your remove the word it's gone!
        self.removed_words = [[self.id2word(bad_id), self.dict_id2word.dfs[bad_id]] for bad_id in bad_ids]

        self.dict_id2word.filter_tokens(bad_ids=bad_ids)


    def remove_single_letters_and_numbers(self):

        self._remove_words_or_IDs(self.single_letters_and_numbers)
        return self

    def remove_stopwords(self):
        self._remove_words_or_IDs(self.standard_stopwords)
        return self

    def remove_custom_stopwords(self, list_of_words):
        """
        A function to allow the removal of custom words from the dictionary.
        :param list_of_words: [str]
                            List of words to be removed from the dictionary.
        :return:
        """
        self.custom_stopwords = set(list_of_words)
        self._remove_words_or_IDs(list_of_words)
        return self

    def remove_common_words(self, percent_of_docs=0.1):
        """
        Remove words that appear in more than `percent_of_docs`
        :param percent_of_docs: float
                                A number between 0 and 1
                                1 means no words are removed.
                                0.5 means words which appear in more than half of the documents will be removed.
                                0 means all words will be removed.
        :return:
        """

        if percent_of_docs > 1 or percent_of_docs < 0:
            raise RuntimeError('percent_of_docs must be between 0 and 1')

        # Further filter words with appear in a given number of documents(jobs).
        self.maxDocFreq = self.dict_id2word.num_docs * percent_of_docs

        logger.info('Ignoring words that appear in documents or more than ' + str(self.maxDocFreq) + ' documents')

        toremove_ids = [tokenid for tokenid, docfreq in self.dict_id2word.dfs.items() if
                        (docfreq > self.maxDocFreq)]
        self._remove_words_or_IDs(toremove_ids)

        return self

    def remove_rare_words(self, percent_of_docs=0.02):

        """
        Remove words that appear in more than `percent_of_docs`
        :param percent_of_docs: float
                                    A number between 0 and 1
                                    1 means all words will be removed.
                                    0.5 means words which appear in more than half of the documents will be removed.
                                    0 no words are removed.
        :return:
        """

        if percent_of_docs > 1 or percent_of_docs < 0:
            raise RuntimeError('percent_of_docs must be between 0 and 1')

        # Further filter words with appear in a given number of documents(jobs).
        self.minDocFreq = self.dict_id2word.num_docs * percent_of_docs

        logger.info('Ignoring words that appear in documents or less than ' + str(self.minDocFreq) + ' documents')

        toremove_ids = [tokenid for tokenid, docfreq in self.dict_id2word.dfs.items() if
                        (docfreq < self.minDocFreq)]
        self._remove_words_or_IDs(toremove_ids)

        return self

    def dataframe_with_document_count_and_removed_words(self):

        # Save dictionary with document frequency for tuning the algorithm later.

        data = [[word_id,
                 self.dict_id2word[word_id],
                 self.dict_id2word.dfs[word_id]] for word_id in self.dict_id2word.keys()]

        DocFreq_df = pd.DataFrame(data)

        DocFreq_df.columns = ['WordID', 'word', 'docfreq']

        set_removedWords = set(self.removed_words)

        DocFreq_df['removed'] = DocFreq_df['WordID'].apply(lambda x: x in set_removedWords)

        return DocFreq_df






