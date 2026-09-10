# Tokenization and Embeddings

A Transformer does not operate on raw characters or whole words. Text is first
split into tokens by a tokenizer, each token is mapped to an integer id, and each
id is mapped to a vector by an embedding table. Everything the model does happens
in that vector space.

## Levels of tokenization

Word-level tokenization produces very large vocabularies, struggles with
morphology, and cannot represent any word it never saw during training; such words
become a single unknown token. Character-level tokenization keeps the vocabulary
tiny and has no unknown tokens, but it makes sequences several times longer and
forces the model to spend capacity relearning spelling. Subword tokenization is
the standard compromise: frequent words stay whole, rare words break into a few
meaningful pieces, and nothing is ever fully unknown because the pieces bottom out
at single characters or bytes.

## Subword algorithms

Byte-pair encoding, or BPE, starts from individual characters and repeatedly
merges the most frequent adjacent pair into a new symbol, building a vocabulary of
common word pieces until a target size is reached. Byte-level BPE, used by GPT-2
and its successors, runs the same procedure over raw bytes, which guarantees every
possible string can be encoded. WordPiece, used by BERT, is similar to BPE but
chooses the merge that most increases the likelihood of the training corpus rather
than the most frequent pair. The unigram language model method, common in
SentencePiece, starts from a large candidate vocabulary and prunes it down by
removing pieces that cost the least likelihood. SentencePiece can also learn
directly from raw text without pre-tokenizing on whitespace, which matters for
languages such as Chinese and Japanese that do not put spaces between words.

## Vocabulary size

Typical vocabulary sizes range from about 30000 tokens for the original BERT, to
roughly 50000 for GPT-2, to 100000 or more for many recent large models. A larger
vocabulary yields shorter token sequences, which speeds up attention because its
cost grows with the square of the sequence length, but it also enlarges the
embedding table and the final output projection, and it means each token is seen
less often during training.

## The embedding table

The embedding layer is a lookup table with one row per vocabulary entry, each row
a trainable vector of size d_model. Turning a token id into its embedding is simply
selecting that row; there is no matrix multiply. These vectors are learned during
training, so tokens that are used in similar ways drift toward similar embeddings.

## Weight tying and scaling

In the original Transformer the same weight matrix is shared three ways: the input
embedding in the encoder, the input embedding in the decoder, and the final linear
layer that projects hidden states to logits over the vocabulary. This weight tying
cuts the parameter count and generally improves quality, since the input and
output token spaces describe the same vocabulary. The paper also multiplies the
embedding vectors by the square root of d_model before the positional encoding is
added, so that the learned embedding signal and the fixed positional signal have
comparable magnitude.

## Static in, contextual out

The vector produced by the embedding table is static: a given token id always
starts as the same vector regardless of its sentence. The context-dependent
meaning of a token, for example the two senses of the word "bank," is built up
only afterward by the stack of attention and feed-forward layers.
