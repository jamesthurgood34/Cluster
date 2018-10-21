"""
Clustering Code

"""

import string

class Cluster(object)

    def __init__(self, input_documents):
        """

        :param input_documents: This should be a dictionary with the key being the ID and the value the string of the document
        """
        self.docs = input_documents

    def remove_punctuation(self):

        to_remove = string.punctuation
        module_logger.info('Removing the following chars: ' + to_remove + '\n')
        replace_punctuation = string.maketrans(to_remove, ' ' * len(to_remove))

        for k in self.docs.keys():
            self.docs[k] = self.docs[k].translate(replace_punctuation)

    def lower_case(self):


