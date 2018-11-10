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

dict = cl.Documents({1: 'this is a sentence',
                  2: 'common abc hjsd sdjksd aka dks asld',
                  3: 'this is',
                  4: 'this this'}).create_dictionary()

dict.print()

dict.remove_custom_stopwords(['dks'])

dict.print()
