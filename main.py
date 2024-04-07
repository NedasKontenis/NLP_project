import keras_nlp
import keras
import json

# Data
SEQ_LEN = 128  # Length of training sequences, in tokens

# Load the trained model
model = keras.models.load_model('C:\\models\\minigpt')

# Load the vocabulary
vocabulary_path = 'C:\\models\\vocabulary\\vocab.json'
with open(vocabulary_path, 'r', encoding='utf-8') as f:
    loaded_vocab = json.load(f)

# Initialize the tokenizer with the loaded vocabulary
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=loaded_vocab,
    sequence_length=SEQ_LEN,
    lowercase=True,
)

# packer adds a start token
start_packer = keras_nlp.layers.StartEndPacker(
    sequence_length=SEQ_LEN,
    start_value=tokenizer.token_to_id("[BOS]"),
)

"""
Inference

We will use the `keras_nlp.samplers` module for inference, which requires a
callback function wrapping the model we just trained. This wrapper calls
the model and returns the logit predictions for the current token we are
generating.

Note: There are two pieces of more advanced functionality available when
defining your callback. The first is the ability to take in a `cache` of states
computed in previous generation steps, which can be used to speed up generation.
The second is the ability to output the final dense "hidden state" of each
generated token. This is used by `keras_nlp.samplers.ContrastiveSampler`, which
avoids repetition by penalizing repeated hidden states. Both are optional, and
we will ignore them for now.
"""
def next(prompt, cache, index):
    logits = model(prompt)[:, index - 1, :]
    # Ignore hidden states for now; only needed for contrastive search.
    hidden_states = None
    return logits, hidden_states, cache

"""
Prompt loop

With our trained model, we can test it out to gauge its performance. To do this
we can seed our model with an input sequence starting with the `"[BOS]"` token,
and progressively sample the model by making predictions for each subsequent
token in a loop.

To start lets build a prompt with the same shape as our model inputs, containing
only the `"[BOS]"` token.
"""

"""
Top-P search

This is where top-p search comes in! Instead of choosing a `k`, we choose a probability
`p` that we want the probabilities of the top tokens to sum up to. This way, we can
dynamically adjust the `k` based on the probability distribution. By setting `p=0.9`, if
90% of the probability mass is concentrated on the top 2 tokens, we can filter out the
top 2 tokens to sample from. If instead the 90% is distributed over 10 tokens, it will
similarly filter out the top 10 tokens to sample from.
"""

while True:
    # Ask for user input for the start prompt
    start_prompt = input("Enter your prompt (or type 'exit' to quit): ")
    if start_prompt.lower() == 'exit':
        break

    # Tokenize the start prompt and pack it, including the [BOS] token.
    prompt_tokens = tokenizer([start_prompt])
    prompt_tokens = start_packer(prompt_tokens)

    sampler = keras_nlp.samplers.TopPSampler(p=0.9)
    output_tokens = sampler(
        next=next,
        prompt=prompt_tokens,
        index=1,
    )
    txt = tokenizer.detokenize(output_tokens)
    print(f"Top-P search generated text: \n{txt}\n")