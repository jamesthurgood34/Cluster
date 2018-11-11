# import Cluster as cl
# import nltk
# emma = nltk.corpus.gutenberg.words('austen-emma.txt')
#
# emma = {k : v for k, v in enumerate(nltk.corpus.gutenberg.raw().split('\n\n'))}
#
# d = cl.Documents(emma)
#
# dict = d.create_dictionary()
#
# dict  = dict.remove_common_words()

import Cluster as cl

dict = cl.Documents( ['this is a sentence',
                  'common abc hjsd hisd sdjksd aka dks asld',
                  'is is',
                  'this this',
                      'words words lots of words this is a long set of words']).create_dictionary()

dict.print()

dict.remove_custom_stopwords(['dks'])

dict.print()

dict = dict.remove_single_letters_and_numbers().remove_stopwords()
