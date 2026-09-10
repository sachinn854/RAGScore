# Training Transformers

Modern Transformer language models are trained in two broad stages, pretraining and
fine-tuning, and sometimes a third alignment stage on top.

## Pretraining

Pretraining runs on a very large corpus of unlabeled text with a self-supervised
objective, meaning the training labels are derived from the data itself rather
than annotated by humans. Decoder-only models such as the GPT family use
next-token prediction, also called causal or autoregressive language modeling:
given all previous tokens, predict the next one. Encoder-only models such as BERT
use masked language modeling, where about 15 percent of the tokens are corrupted
and the model must recover the originals using context from both sides.
Encoder-decoder models such as T5 use a span-corruption objective, where
contiguous spans are replaced by sentinel tokens and the decoder reconstructs
them.

In every case the loss function is cross-entropy between the model's predicted
probability distribution over the vocabulary and the actual token, averaged over
all predicted positions. Minimizing cross-entropy is equivalent to maximizing the
likelihood the model assigns to the training text, and it is often reported as
perplexity, which is the exponential of the average loss.

## Fine-tuning and alignment

Fine-tuning continues training the pretrained model on a smaller labeled dataset
for a specific task such as classification, extractive question answering, or
summarization. Because the model already carries broad language knowledge from
pretraining, fine-tuning needs far less data and compute than training from
scratch. Instruction tuning is fine-tuning on many tasks phrased as natural
language instructions, which teaches the model to follow prompts. A further stage,
reinforcement learning from human feedback, trains a reward model from human
preference comparisons and then optimizes the language model against that reward to
make outputs more helpful and safe.

## The optimizer and learning-rate schedule

The original Transformer paper trains with the Adam optimizer using beta1 of 0.9,
beta2 of 0.98, and epsilon of 1e-9. The learning rate is not constant. It rises
linearly for the first 4000 warmup steps and then decays in proportion to the
inverse square root of the step number. The warmup avoids large, destabilizing
updates early in training when the parameters are still near their random
initialization and the Adam variance estimates are unreliable. Later models often
use a linear or cosine decay to a small final value instead.

## Regularization

Several techniques are used together. Dropout with rate 0.1 is applied throughout
the base network. Label smoothing of 0.1 is applied to the targets, so the model is
trained to put probability 0.9 on the correct token and spread the remaining 0.1
across the rest of the vocabulary. Label smoothing slightly worsens perplexity but
improves accuracy and BLEU because it stops the model from becoming overconfident.
Weight decay and gradient clipping are also common in later recipes.

## Scale and hardware

Large models are trained across many GPUs or TPUs in parallel, using data
parallelism, tensor parallelism, and pipeline parallelism together. Batches are
usually measured in tokens rather than sentences, often hundreds of thousands of
tokens per step, so that each step holds a similar amount of computation
regardless of individual sequence lengths. Empirical scaling laws show that test
loss falls predictably as a power law in model size, dataset size, and compute,
which is why model and data sizes have grown together.
