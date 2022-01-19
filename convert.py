from gensim.scripts.glove2word2vec import glove2word2vec

(count, dimensions) = glove2word2vec(".vector_cache/glove/glove.42B.300d.txt", "cropus/42B_w2v.txt")