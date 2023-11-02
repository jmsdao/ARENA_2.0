import os, sys
from pathlib import Path
chapter = r"chapter1_transformers"
for instructions_dir in [
    Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/instructions").resolve(),
    Path("/app/arena_2.0/chapter1_transformers/instructions").resolve(),
    Path("/mount/src/arena_2.0/chapter1_transformers/instructions").resolve(),
]:
    if instructions_dir.exists():
        break
if str(instructions_dir) not in sys.path: sys.path.append(str(instructions_dir))
os.chdir(instructions_dir)

import streamlit as st
import st_dependencies

st_dependencies.styling()

import platform
is_local = (platform.processor() != "")

import streamlit_analytics
streamlit_analytics.start_tracking()

def section_0_july():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li><a class='contents-el' href='#prerequisites'>Prerequisites</a></li>
    <li><a class='contents-el' href='#motivation'>Motivation</a></li>
    <li><a class='contents-el' href='#logistics'>Logistics</a></li>
    <li><a class='contents-el' href='#what-counts-as-a-solution'>What counts as a solution?</a></li>
    <li><a class='contents-el' href='#setup'>Setup</a></li>
    <li><a class='contents-el' href='#task-dataset'>Task & Dataset</a></li>
    <li><a class='contents-el' href='#model'>Model</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(
r"""
# Monthly Algorithmic Challenge (July 2023): Palindromes

### Colab: [problem](https://colab.research.google.com/drive/1qTUBj16kp6ZOCEBJefCKdzXvBsU1S-yz) | [solutions](https://colab.research.google.com/drive/1zJepKvgfEHMT1iKY3x_CGGtfSR2EKn40)

This marks the first of the (hopefully sequence of) monthly mechanistic interpretability challenges. I designed them in the spirit of [Stephen Casper's challenges](https://www.lesswrong.com/posts/KSHqLzQscwJnv44T8/eis-vii-a-challenge-for-mechanists), but with the more specific aim of working well in the context of the rest of the ARENA material, and helping people put into practice all the things they've learned so far.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/zoom.png" width="350">

## Prerequisites

The following ARENA material should be considered essential:

* **[1.1] Transformer from scratch** (sections 1-3)
* **[1.2] Intro to Mech Interp** (sections 1-3)

The following material isn't essential, but is very strongly recommended:

* **[1.2] Intro to Mech Interp** (section 4)
* **[1.4] Balanced Bracket Classifier** (all sections)

## Motivation

Neel Nanda's post [200 COP in MI: Interpreting Algorithmic Problems](https://www.lesswrong.com/posts/ejtFsvyhRkMofKAFy/200-cop-in-mi-interpreting-algorithmic-problems) does a good job explaining the motivation behind solving algorithmic problems such as these. I'd strongly recommend reading the whole post, because it also gives some high-level advice for approaching such problems.

The main purpose of these challenges isn't to break new ground in mech interp, rather they're designed to help you practice using & develop better understanding for standard MI tools (e.g. interpreting attention, direct logit attribution), and more generally working with libraries like TransformerLens.

Also, they're hopefully pretty fun, because why shouldn't we have some fun while we're learning?

## Logistics

If this first problem is well-received, I'll try to post a new one every month. Because I think this one is on the easier side relatively speaking, I'll leave it open until the end of July (which at time of writing is 16 days). **My solution will be published on 31st July on this page**, at the same time as the next problem in the sequence. Future challenges will also be accompanied by a LessWrong post, but not this one (because it's experimental).

If you try to interpret this model, you can send your attempt in any of the following formats:

* Colab notebook
* GitHub repo (e.g. with ipynb or markdown file explaining results)
* Google Doc (with screenshots and explanations)
* or any other sensible format.

You can send your attempt to me (Callum McDougall) via any of the following methods:

* The [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), via a direct message to me
* My personal email: `cal.s.mcdougall@gmail.com`
* LessWrong message ([here](https://www.lesswrong.com/users/themcdouglas) is my user)

**I'll feature the names of everyone who sends me a solution on this website, and also give a shout out to the best solutions.**

Please don't discuss specific things you've found about this model until the challenge is over (although you can discuss general strategies and techniques, and you're also welcome to work in a group if you'd like). The deadline for this problem will be the end of this month, i.e. 31st July. If the challenge is well-received (which I'm arbitrarily defining as there being at least 5 submissions which I judge to be high-quality), then I'll make it a monthly sequence.

## What counts as a solution?

Going through the exercises **[1.4] Balanced Bracket Classifier** should give you a good idea of what I'm looking for. This model is much less complicated than the one in that exercise, so I'd have a higher standard for what counts as a full solution. In particular, I'd expect you to:

* Describe a mechanism for how the model solves the task, in the form of the QK and OV circuits of various attention heads (and possibly any other mechanisms the model uses, e.g. the direct path, or nonlinear effects from layernorm),
* Provide evidence for your mechanism, e.g. with tools like attention plots, targeted ablation / patching, or direct logit attribution.
* (Optional) Include additional detail, e.g. identifying the linear subspaces that the model uses for certain forms of information transmission.

# Setup

```python
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from pathlib import Path

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "monthly_algorithmic_problems" / "july23_palindromes"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from monthly_algorithmic_problems.july23_palindromes.dataset import PalindromeDataset, display_seq
from monthly_algorithmic_problems.july23_palindromes.model import create_model
from plotly_utils import hist, bar, imshow

device = t.device("cuda" if t.cuda.is_available() else "cpu")
```

## Task & Dataset

The directory containing all the relevant files is `chapter1_transformers/exercises/monthly_algorithmic_problems/july23_palindromes`. This contains files `model.py` (for defining the model), `training.py` (for training the model), and `dataset.py` (for the dataset of palindromes and non-palindromes).

Each sequence in the dataset looks like:

```
[start_token, a_1, a_2, ..., a_N, end_token]
```

where `start_token = 31`, `end_token = 32`, and each value `a_i` is a value in the range `[0, 30]` inclusive. 

Each sequence has a corresponding label, which is `1` if the sequence is a palindrome (i.e. `(a_1, a_2, ..., a_N) == (a_N, ..., a_2, a_1)`), and `0` otherwise. The model has been trained to classify each sequence according to this label.

We've given you the class `PalindromeDataset` to store your data. You can slice this object to get batches of tokens and labels. You can also use the function `display_seq` to display a sequence in a more readable format (with any tokens that stop it from being a palindrome highlighted). There's an example later on this page. 

Some other useful methods and attributes of this dataset (you can inspect `dataset.py` to see for yourself) are:

* `dataset.toks`, to get a batch of all the tokens in the dataset, of shape `(size, 2 * half_length + 2)`.
* `dataset.is_palindrome`, to get a tensor of all the labels in the dataset, of shape `(size,)`.
* `dataset.str_toks`, to get a list of lists, with string representations of each sequence, e.g. `["START", "1", "4", ..., "END"]`. This is useful for visualisation, e.g. circuitsvis.

## Model

Our model was trained by minimising cross-entropy loss between its predictions and the true labels. You can inspect the notebook `training_model.ipynb` to see how it was trained.

The model is is a 2-layer transformer with 2 attention heads, and causal attention. It includes layernorm, but no MLP layers. You can load it in as follows:

```python
filename = section_dir / "palindrome_classifier.pt"

model = create_model(
    half_length=10, # this is half the length of the palindrome sequences
    max_value=30, # values in palindrome sequence are between 0 and max_value inclusive
    seed=42,
    d_model=28,
    d_head=14,
    n_heads=2,
    normalization_type="LN",
    d_mlp=None # this is an attn-only model
)

state_dict = t.load(filename)

state_dict = model.center_writing_weights(t.load(filename))
state_dict = model.center_unembed(state_dict)
state_dict = model.fold_layer_norm(state_dict)
state_dict = model.fold_value_biases(state_dict)
model.load_state_dict(state_dict, strict=False);
```

The code to process the state dictionary is a bit messy, but it's necessary to make sure the model is easy to work with. For instance, if you inspect the model's parameters, you'll see that `model.ln_final.w` is a vector of 1s, and `model.ln_final.b` is a vector of 0s (because the weight and bias have been folded into the unembedding).

```python
print("ln_final weight: ", model.ln_final.w)
print("\nln_final, bias: ", model.ln_final.b)
```

<details>
<summary>Aside - the other weight processing parameters</summary>

Here's some more code to verify that our weights processing worked, in other words:

* The unembedding matrix has mean zero over both its input dimension (`d_model`) and output dimension (`d_vocab`)
* All writing weights (i.e. `b_O`, `W_O`, and both embeddings) have mean zero over their output dimension (`d_model`)
* The value biases `b_V` are zero (because these can just be folded into the output biases `b_O`)

```python
W_U_mean_over_input = einops.reduce(model.W_U, "d_model d_vocab -> d_model", "mean")
t.testing.assert_close(W_U_mean_over_input, t.zeros_like(W_U_mean_over_input))

W_U_mean_over_output = einops.reduce(model.W_U, "d_model d_vocab -> d_vocab", "mean")
t.testing.assert_close(W_U_mean_over_output, t.zeros_like(W_U_mean_over_output))

W_O_mean_over_output = einops.reduce(model.W_O, "layer head d_head d_model -> layer head d_head", "mean")
t.testing.assert_close(W_O_mean_over_output, t.zeros_like(W_O_mean_over_output))

b_O_mean_over_output = einops.reduce(model.b_O, "layer d_model -> layer", "mean")
t.testing.assert_close(b_O_mean_over_output, t.zeros_like(b_O_mean_over_output))

W_E_mean_over_output = einops.reduce(model.W_E, "token d_model -> token", "mean")
t.testing.assert_close(W_E_mean_over_output, t.zeros_like(W_E_mean_over_output))

W_pos_mean_over_output = einops.reduce(model.W_pos, "position d_model -> position", "mean")
t.testing.assert_close(W_pos_mean_over_output, t.zeros_like(W_pos_mean_over_output))

b_V = model.b_V
t.testing.assert_close(b_V, t.zeros_like(b_V))
```

</details>

The model was trained to output the correct classification at the `END` token, in other words the value of the residual stream at `END` (post-layernorm) is mapped through `model.W_U` which has shape `(d_model, 2)`, and this gives us our classification logits for `(not palindrome, palindrome)`.

A demonstration of the model working (and of the `display_seq` function):

```python
dataset = PalindromeDataset(size=100, max_value=30, half_length=10)

toks, is_palindrome = dataset[:5]

logits = model(toks)[:, -1]
probs = logits.softmax(-1)
probs_palindrome = probs[:, 1]

for tok, prob in zip(toks, probs_palindrome):
    display_seq(tok, prob)
```

<details>
<summary>Click on this dropdown for a hint on how to start (and some example code).</summary>

The following code will display the attention patterns for each head, on a particular example.

```python
display_seq(dataset.toks[batch_idx], probs_palindrome[batch_idx])

import circuitsvis as cv

cv.attention.attention_patterns(
    attention = t.concat([cache["pattern", layer][batch_idx] for layer in range(model.cfg.n_layers)]),
    tokens = dataset.str_toks[batch_idx],
    attention_head_names = [f"{layer}.{head}" for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)],
)
```

Find (1) a palindromic example, and (2) a non-palindromic example which is close to being palindromic (i.e. only 1 or 2 tokens are different). Then, compare the attention patterns for these two examples. Questions you might want to answer:

* How do the attention patterns for numbers which are palindromic (i.e. they are the same as their mirror image) differ from the numbers which aren't?
* How does information eventually get to the `[END]` token?

</details>

Note - although this model was trained for long enough to get loss close to zero (you can test this for yourself), it's not perfect. There are some weaknesses that the model has which make it vulnerable to adversarial examples, which I've decided to leave in as a fun extra challenge! Note that the model is still very good at its intended task, and the main focus of this challenge is on figuring out how it solves the task, not dissecting the situations where it fails. However, you might find that the adversarial examples help you understand the model better.

Best of luck! 🎈

""", unsafe_allow_html=True)
    
palindromes_dir = instructions_dir / "media/palindromes"
unique_char_dir = instructions_dir / "media/unique_char"
sum_dir = instructions_dir / "media/sum"
sorted_list_dir = instructions_dir / "media/sorted_list"
cumsum_dir = instructions_dir / "media/cumsum"
import plotly.graph_objects as go
from streamlit.components.v1 import html as st_html
import json

def section_1_july():
    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li><a class='contents-el' href='#0-hypotheses'>0. Hypotheses</a></li>
    <li><a class='contents-el' href='#1-eyeball-attention-patterns'>1. Eyeball attention patterns</a></li>
    <li><a class='contents-el' href='#2-head-ablations'>2. Head ablations</a></li>
    <li><a class='contents-el' href='#3-full-qk-matrix-of-head-0-0'>3. Full QK matrix of head <code>0.0</code></a></li>
    <li><a class='contents-el' href='#4-investigating-adversarial-examples'>4. Investigating adversarial examples</a></li>
    <li><a class='contents-el' href='#5-composition-of-0-0-and-1-0'>5. Composition of <code>0.0</code> and <code>1.0</code></a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#first-experiment-k-composition'>K-composition</a></li>
        <li><a class='contents-el' href='#second-experiment-v-composition'>V-composition</a></li>
    </ul></li>
    <br>
    <li><a class='contents-el' href='#a-few-more-experiments'>A few more experiments</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#targeted-ablation'>Targeted ablations</a></li>
        <li><a class='contents-el' href='#composition-scores'>Composition scores</a></li>
        <li><a class='contents-el' href='#how-is-the-non-palindromic-information-stored'>How is the "non-palindromic" information stored?</a></li>
    </ul></li>
</ul></li>""", unsafe_allow_html=True)
    
    st.markdown(
r"""
# Monthly Algorithmic Challenge (July 2023): Solutions

We assume you've run all the setup code from the previous page "[July] Palindromes". Here's all the new setup code you'll need:

```python
dataset = PalindromeDataset(size=2500, max_value=30, half_length=10).to(device)

logits, cache = model.run_with_cache(dataset.toks)

logprobs = logits[:, -1].log_softmax(-1)
probs = logprobs.softmax(-1)
probs_palindrome = probs[:, 1]

logprobs_correct = t.where(dataset.is_palindrome.bool(), logprobs[:, 1], logprobs[:, 0])
logprobs_incorrect = t.where(dataset.is_palindrome.bool(), logprobs[:, 0], logprobs[:, 1])
probs_correct = t.where(dataset.is_palindrome.bool(), probs[:, 1], probs[:, 0])

avg_logit_diff = (logprobs_correct - logprobs_incorrect).mean().item()
avg_cross_entropy_loss = -logprobs_correct.mean().item()
print(f"Average cross entropy loss: {avg_cross_entropy_loss:.3f}")
print(f"Average logit diff: {avg_logit_diff:.3f}")
```

<div style='font-family:monospace; font-size:15px;'>
Average cross entropy loss: 0.008<br>
Average logit diff: 7.489
</div><br>

Denote the vectors in the residual stream (other than `START` and `END`) as $\{x_1, x_2, ..., x_{20}\}$. Each $x_i = t_i + p_i$ (the token embedding plus positional embedding). We say that the $i$-th token is palindromic if $t_i = t_{20-i}$ (so the whole sequence is palindromic if and only if all $x_{11}, ..., x_{20}$ are palindromic). We'll sometimes use $x$ to refer to a token in the second half of the sequence, and $x'$ to that token's mirror image.

Rather than going for a "rational reconstruction", I've tried to present the evidence roughly in the order I found it, so this should give you one perspective on what the mech interp process can look like.

# 0. Hypotheses

It's a good idea to spend some time brainstorming hypotheses about how the model might go about solving the task. After thinking about it for a while, I came up with the following two hypotheses:

**1. Reflection**

Each token $x$ in the second half will attend back to $x'$ to get information about whether the two are equal. This information then gets moved to the `END` token.

If this is true, then I expect to see one or both of the layer-0 heads attending in a "lambda pattern" (thanks to Andy Arditi for this terminology), i.e. 20 attends to 1, 19 attends to 2, etc. In layer 1, I expect to see the `END` token attending to the tokens in the second half, where this information is stored. In particular, in the non-palindromic cases I expect to see `END` attending to the tokens in the second half which are non-palindromic (because it only takes one unbalanced pair for the sequence to be non-palindromic). We might expect `END` to attend to the `START` token in palindromic sequences, since it's a useful rest position.

**2. Aggregation**

The `END` token (or the last non-END token) attends uniformly to all tokens, and does some kind of aggregation like the brackets task (i.e. it stores information about whether each token is equal to its reflection). Then, nonlinear operations on the `END` token (self-attention from layer 1 and softmax) turn this aggregated information into a classification.

**Evaluation of these two hypotheses**

Aggregation seems much less likely, because it's not making use of any of the earlier sequence positions to store information, and it's also not making use of the model's QK circuit (i.e. half the model). Maybe aggregation would be more likely if we had MLPs, but not for an attention-only model. Reflection seems like a much more natural hypothesis.

# 1. Eyeball attention patterns

Both the hypotheses above would be associated with very distinctive attention patterns, which is why plotting attention patterns is a good first step here.

I've used my own circuitsvis code which sets up a selection menu to view multiple patterns at once, and I've also used a little HTML hacking to highlight the tokens which aren't palindromic (this is totally unnecessary, but made things a bit visually clearer for me!).

```python
def red_text(s: str):
    return f"<span style='color:red'>{s}</span>"


def create_str_toks_with_html(toks: Int[Tensor, "batch seq"]):
    '''
    Creates HTML which highlights the tokens that don't match their mirror images. Also puts a gap 
    between each token so they're more readable.
    '''
    raw_str_toks = [["START"] + [f"{t:02}" for t in tok[1:-1]] + ["END"] for tok in toks]

    toks_are_palindromes = toks == toks.flip(-1)
    str_toks = []
    for raw_str_tok, palindromes in zip(raw_str_toks, toks_are_palindromes):
        str_toks.append([
            "START - ", 
            *[f"{s} - " if p else f"{red_text(s)} - " for s, p in zip(raw_str_tok[1:-1], palindromes[1:-1])], 
            "END"
        ])
    
    return str_toks


cv.attention.from_cache(
    cache = cache,
    tokens = create_str_toks_with_html(dataset.toks),
    batch_idx = list(range(10)),
    attention_type = "info-weighted",
    radioitems = True,
)
```
""", unsafe_allow_html=True)
    
    with open(palindromes_dir / "fig1.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=625)

    st.markdown(r"""
## Conclusions

* The reflection hypotheses seems straightforwardly correct.
* Head 0.0 is impelenting the "lambda pattern".
    * We can see that $x$ attends back to $x'$ if they're the same, otherwise it usually self-attends.
    * This suggests the quantity $(x - x')^T W_{OV}^{0.0}$ might be important (this is the difference between the vector which is added at $x$ when $x$ is palindromic vs. non-palindromic, ignoring layernorm and assuming attention is always either 0 or 1). I'll return to this later.
* Head 0.1 isn't really doing this, or anything distinctive - maybe this head isn't important?
    * Head 0.1 is actually doing something important at $x = 20$, but I didn't spot this at the time.
* Head 1.0 is implementing the "attend to non-palindromic tokens in the second half" pattern.
    * Although one part of my hypothesis was false - `START` doesn't seem like it's used as an attention placeholder for palindromic sequences.
    * The attention patterns from `END` to other tokens seem pretty random in palindromic sequences.
    * This suggests we might be seeing V-composition between heads 0.0 and 1.0 (otherwise the signal which 1.0 is picking up on in non-palindromic sequences would also be picked up in palindromic sequences, and the model wouldn't work).
* Head 1.1 is attending to $x_{20}$ when it's non-palindromic.
    * Maybe it's doing this to compensate for head 1.0, which never seems to attend to $x_{20}$.

## Other notes

* Using info-weighted attention is a massive win here. In particular, it makes the behaviour of head 1.0 a lot clearer than just using regular attention.

## Next experiments to run

* I think 0.1 and 1.1 are unimportant - to test this I should ablate them and see if loss changes. If not, then I can zoom in on 0.0 and 1.0 for most of the rest of my analysis.
* 0.0 seems to be implementing a very crisp attention pattern - I should look at the full QK circuit to see how this is implemented.
* After these two experiments (assuming the evidence from them doesn't destroy any of my current hypotheses), I should try and investigate how 0.0 and 1.0 are composing.

# 2. Head ablations

I want to show that heads 0.1 and 1.1 don't really matter, so I'm going to write code to ablate them and see how the loss changes.

Note, I'm ablating the head's result vector (because this makes sure we ablate both the QK and OV circuit signals). On larger models we might have to worry about storing `result` in our cache, but this is a very small model so we don't need to worry about that here.
                
```python
def get_loss_from_ablating_head(layer: int, head: int, ablation_type: Literal["zero", "mean"]):

    def hook_patch_result_mean(result: Float[Tensor, "batch seq nheads d_model"], hook: HookPoint):
        '''
        Ablates an attention head (either mean or zero ablation).
        
        Note, when mean-ablating we don't average over sequence positions. Can you see why this is important?
        (You can return here after you understand the full algorithm implemented by the model.)
        '''
        if ablation_type == "mean":
            result_mean: Float[Tensor, "d_model"] = cache["result", layer][:, :, head].mean(0)
            result[:, :, head] = result_mean
        elif ablation_type == "zero":
            result[:, :, head] = 0
        return result

    model.reset_hooks()
    logits = model.run_with_hooks(
        dataset.toks,
        fwd_hooks = [(utils.get_act_name("result", layer), hook_patch_result_mean)],
    )[:, -1]
    logits_correct = t.where(dataset.is_palindrome.bool(), logits[:, 1], logits[:, 0])
    logits_incorrect = t.where(dataset.is_palindrome.bool(), logits[:, 0], logits[:, 1])
    avg_logit_diff = (logits_correct - logits_incorrect).mean().item()
    return avg_logit_diff
    


print(f"Original logit diff = {avg_logit_diff:.3f}")

for ablation_type in ["mean", "zero"]:
    print(f"\nNew logit diff after {ablation_type}-ablating head...")
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            print(f"...{layer}.{head} = {get_loss_from_ablating_head(layer, head, ablation_type):.3f}")
```

<div style='font-family:monospace; font-size:15px;'>
Original logit diff = 7.489<br><br>
New logit diff after mean-ablating head...<br>
...0.0 = 0.614<br>
...0.1 = 6.642<br>
...1.0 = 0.299<br>
...1.1 = 6.815<br><br>
New logit diff after zero-ablating head...<br>
...0.0 = 1.672<br>
...0.1 = 3.477<br>
...1.0 = 2.847<br>
...1.1 = 7.274<br>
</div><br>

Mean ablation shows us that heads 0.1 and 1.1 aren't crucial. Interestingly, zero-ablation would lead us to believe (incorrectly) that head 0.1 is very important. This is a common problem, especially with early heads (because zero-ablating these heads output will be moving the input of later heads off-distribution).

At this point I thought that 1.1 was doing something important at position 20, but decided not to investigate it yet, because looking more into 0.0 and 1.0 seemed like it should tell me most of what I wanted to know about this model.

# 3. Full QK matrix of head 0.0

I wanted to see what the full QK matrices of the heads looked like. I generated them for both heads in layer 0, and also for heads in layer 1 (but I guessed these wouldn't tell me as much, because composition would play a larger role in these heads' input, hence I don't show the layer-1 plots below).

In the attention scores plot, I decided to concatenate the embedding and positional embedding matrices, so I could see all interactions between embeddings and positional embeddings. The main reason I did this wasn't for the cross terms (I didn't expect to learn much from seeing how much token $t_i$ attends to position $p_j$), but just so that I could see all the $(t_i, t_j)$ terms next to the $(p_i, p_j)$ terms in a single plot (and compare them to see if positions or tokens had a larger effect on attention scores).

```python
W_QK: Float[Tensor, "layers heads d_model d_model"] = model.W_Q @ model.W_K.transpose(-1, -2)

W_E_pos = t.concat([model.W_E, model.W_pos], dim=0)

W_QK_full = W_E_pos @ W_QK @ W_E_pos.T

d_vocab = model.cfg.d_vocab
n_ctx = model.cfg.n_ctx
assert W_QK_full.shape == (2, 2, d_vocab + n_ctx, d_vocab + n_ctx)

# More use of HTML to increase readability - plotly supports some basic HTML for titles and axis labels
W_E_labels = [f"W<sub>E</sub>[{i}]" for i in list(range(d_vocab - 2)) + ["START", "END"]]
W_pos_labels = [f"W<sub>pos</sub>[{i}]" for i in ["START"] + list(range(1, n_ctx - 1)) + ["END"]]

imshow(
    W_QK_full.flatten(0, 1),
    title = "Full QK matrix for different heads (showing W<sub>E</sub> and W<sub>pos</sub>)",
    x = W_E_labels + W_pos_labels,
    y = W_E_labels + W_pos_labels,
    labels = {"x": "Source", "y": "Dest"},
    facet_col = 0,
    facet_labels = ["0.0", "0.1", "1.0", "1.1"],
    height = 1000,
    width = 1900,
)
```
""", unsafe_allow_html=True)
    
    fig2 = go.Figure(json.loads(open(palindromes_dir / "fig2.json", 'r').read()))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(r"""
## Conclusions

* As expected, 0.0 had the most distinctive patterns.
    * The strongest pattern was in the token embeddings - each token attends to itself
    * There was a weaker pattern in the positional embeddings - each position in the second half attends to either itself or its mirror (usually with a preference for its mirror)
    * Combining these 2 heuristics, we can see the algorithm that is implemented is the same as the pattern we observed earlier:
        * *If $x$ and $x'$ are the same, then $x$ attends back to $x'$. Otherwise, $x$ self-attends.*
* Head 0.1 has pretty clear patterns too.
    * Oddly, each token anti self-attends.
    * Position 20 (and 18 to a lesser extent) have high attention scores to themselves and their mirror position.
    * Combining these two pieces of evidence, I would guess that 0.1 is doing the same thing as 0.0 is doing, but in reverse, and only for the token pairs $(1, 20)$ and $(3, 18)$:
        * *If $x$ and $x'$ are different, then $x$ attends back to $x'$. Otherwise, $x$ attends to both $x$ and $x'$.*
    * I would guess that this signal is being used in the same way as the signal from 0.0, but in the opposite direction.
    * Going back to the attention patterns at the top, we can see that this is happening for token 20 (not for 18).

To make the plots for head 0.0 clearer, I plotted $W_E$ and $W_{pos}$ for head 0.0 separately, and after softmaxing (note I apply a causal mask for positions, not for tokens). Because I wanted to see if any nonlinear trickery was happening with the layernorm in layer zero, I checked the standard deviation of layernorm at each sequence position - it was very small, meaning this kind of plot is reasonable.

Note that applying softmax can sometimes be misleading, because it extremises patterns and risks making them appear too clean. 

```python
# Check layernorm scale factor mean & std dev, verify that std dev is small
scale = cache["scale", 0, "ln1"][:, :, 0, 0] # shape (batch, seq)
df = pd.DataFrame({
    "std": scale.std(0).cpu().numpy(),
    "mean": scale.mean(0).cpu().numpy(),
})
px.bar(
    df, 
    title="Mean & std of layernorm before first attn layer", 
    template="simple_white", width=600, height=400, barmode="group"
).show()


# Get full matrix for tokens (we take mean over entire LN scale)
W_QK = model.W_Q[0, 0] @ model.W_K[0, 0].T / (model.cfg.d_head ** 0.5)
W_E_scaled = model.W_E[:-2] / scale.mean()
W_QK_full_tokens = W_E_scaled @ W_QK @ W_E_scaled.T

# Get full matrix for tokens (here, we can preserve the seq dim in `scale`)
W_pos_scaled = model.W_pos[1:-1] / scale[:, 1:-1].mean(dim=0).unsqueeze(-1)
# Scale by sqrt(d_head)
W_QK_full_pos = W_pos_scaled @ W_QK @ W_pos_scaled.T
# Apply causal mask 
W_QK_full_pos.masked_fill_(~t.tril(t.ones_like(W_QK_full_pos)).bool(), -1e6)

# Plot both
for (name, matrix) in zip(["tokens", "positions"], [W_QK_full_tokens, W_QK_full_pos]):
    imshow(
        matrix.softmax(-1),
        title = f"Full QK matrix for 0.0 ({name})",
        x = W_pos_labels[1:-1] if name == "positions" else W_E_labels[:-2],
        y = W_pos_labels[1:-1] if name == "positions" else W_E_labels[:-2],
        labels = {"x": "Source", "y": "Dest"},
        height = 800,
        width = 800,
    )
```
""", unsafe_allow_html=True)
    
    fig3 = go.Figure(json.loads(open(palindromes_dir / "fig3.json", 'r').read()))
    fig4_tokens = go.Figure(json.loads(open(palindromes_dir / "fig4_tokens.json", 'r').read()))
    fig4_positions = go.Figure(json.loads(open(palindromes_dir / "fig4_positions.json", 'r').read()))
    st.plotly_chart(fig3, use_container_width=False)
    st.plotly_chart(fig4_tokens, use_container_width=False)
    st.plotly_chart(fig4_positions, use_container_width=False)

    st.markdown(r"""
Result - we can clearly see the pattern that was observed earlier. However, some results aren't as clean as I was expecting (in particular the positional results). The blind spots at positions 17 and 19 are very apparent here.

# 4. Investigating adversarial examples

Before looking at composition between 0.0 and 1.0, I'm going to take a look at the blind spots at 17 and 19, and see if I can generate some adversarial examples. I'll construct strings where only one pair is non-palindromic, and look at the classification probabilities.

```python
# Pick a string to start with, check it's palindromic
batch_idx = 1
assert dataset.is_palindrome[batch_idx].item() == 1

# Create my adversarial examples (with some non-adversarial examples as a baseline)
test_toks = {None: dataset.toks[batch_idx].clone()}
for i in [17, 18, 19, 20]:
    test_toks[i] = dataset.toks[batch_idx].clone()
    test_toks[i][i] += 1
test_toks = t.stack(list(test_toks.values()))

test_logits, test_cache = model.run_with_cache(test_toks)
test_probs = test_logits[:, -1].softmax(-1)
test_probs_balanced = test_probs[:, 1]

for k, v in zip([None, 17, 18, 19, 20], test_probs_balanced):
    print(f"{k} flipped, P(palindrome) = {v:.3f}")

cv.attention.from_cache(
    cache = test_cache,
    tokens = create_str_toks_with_html(test_toks),
    attention_type = "info-weighted",
    radioitems = True,
)
```
""", unsafe_allow_html=True)
    
    with open(palindromes_dir / "fig5.html", 'r') as f: fig5 = f.read()
    st_html(fig5, height=525)

    st.markdown(r"""
## Conclusion

This is exactly what I expected - 17 and 19 are adversarial examples. When only one of these positions is non-palindromic, the model will incorrectly classify the sequence as palindromic with high probability.

We can investigate further by looking at all the advexes in the dataset, and seeing how many of them are of this form. The results show that 2/3 of the "natural advexes" are of this form. Also, every single one of the "type 17/19 sequences" (i.e. the ones which are only non-palindromic at positions 17 or 19) are advexes.

<details>
<summary>A note on why these advexes exist</summary>

The way non-palindromic sequences are generated in the dataset is as follows: a random subset of tokens in the second half are chosen to be non-palindromic, with the size of this subset having a $\operatorname{Binomial}(10, 1/2)$ distribution (i.e. each token was randomly chosen to be palindromic or non-palindromic). This means that, for any small subset, the probability that a sequence is only non-palindromic within that subset is pretty small - hence adversarial examples can easily form.

Two exercises to the reader:

* What is the probability of a sequence generated in this way being non-palindromic only within the subset $\{17, 19\}$?
* How could you change the data generation process to make it harder for adversarial examples like these to form?

</details>

```python
is_advex = (probs_correct < 0.5)

is_palindromic_per_token = (dataset.toks == dataset.toks.flip(-1))
advex_indices = [17, 19]
non_advex_indices = [i for i in range(11, 21) if i not in advex_indices]

is_palindrome_at_non_advex = t.all(is_palindromic_per_token[:, non_advex_indices], dim=-1)
is_17_or_19_type = is_palindrome_at_non_advex & t.any(~is_palindromic_per_token[:, advex_indices], dim=-1)

print(f"Number of advexes which are in the 17/19 category:    {(is_17_or_19_type & is_advex).sum()}")
print(f"Number of advexes which aren't in the 17/19 category: {(~is_17_or_19_type & is_advex).sum()}")
print(f"Number of type-17/19 which aren't advexes:            {(is_17_or_19_type & ~is_advex).sum().item()}")

print("\nAdversarial examples:")
from IPython.display import display, HTML
display(HTML("<br>".join(["".join(x) for x in create_str_toks_with_html(dataset.toks[is_advex])])))
```

Result:

<div style='font-family:times-new-roman;'>
START - 13 - <span style='color:red'>30</span> - 18 - <span style='color:red'>17</span> - 25 - 23 - 11 - 24 - 09 - 01 - 01 - 09 - 24 - 11 - 23 - 25 - <span style='color:red'>20</span> - 18 - <span style='color:red'>11</span> - 13 - END<br>START - <span style='color:red'>27</span> - 12 - 23 - 27 - 21 - 25 - 24 - 24 - 25 - 15 - 15 - 25 - 24 - 24 - 25 - 21 - 27 - 23 - 12 - <span style='color:red'>23</span> - END<br>START - 23 - <span style='color:red'>05</span> - 06 - <span style='color:red'>02</span> - 24 - 18 - 18 - 13 - 19 - 23 - 23 - 19 - 13 - 18 - 18 - 24 - <span style='color:red'>05</span> - 06 - <span style='color:red'>30</span> - 23 - END
</div>

<br>

# 5. Composition of 0.0 and 1.0

This is the final big question that needs to be answered - how are `0.0` and `1.0` composing to give us the actual result?

Here, we return to the quantity $(x - x')^T W_{OV}^{0.0}$ discussed earlier, and I justify my choice of this vector.

Suppose each $x$ attends to $(x, x')$ with probability $(p_1, p_1')$ respectively when $x$ is palindromic, and $(p_2, p_2')$ when $x$ is non-palindromic (so we expect $p_1 + p_1' \approx 1, p_2 + p_2' \approx 1$ in most cases, and $p_2 > p_1$). This means that the vector added to $x$ is $p_2 x^T W_{OV}^{0.0} + p_2' x'^T W_{OV}^{0.0}$ in the non-palindromic case, and $p_1 x^T W_{OV}^{0.0} + p_1' x'^T W_{OV}^{0.0}$ in the palindromic case. The difference between these two vectors is:

$$
((p_2 - p_1) x - (p_1' - p_2') x')^T W_{OV}^{0.0} \approx (p_2 - p_1) (x - x')^T W_{OV}^{0.0}
$$

where I've used the approximations $p_1 + p_1' \approx 1, p_2 + p_2' \approx 1$. This is a positive mulitple of the thing we've defined as our "difference vector". Therefore, it's natural to guess that the "this token is non-palindromic" information is stored in the direction defined by this vector.

First, we should check that both $p_2 - p_1$ and $p_1' - p_2'$ are consistently positive (this definitely looked like the case when we eyeballed attention patterns, but we'd ideally like to be more careful).

Note - the plot that I'm making here is a box plot, which I don't have code for in `plotly_utils`. When there's a plot like this which I find myself wanting to make, I usually defer to using ChatGPT (creating quick and clear visualisations is one of the main ways I use it in my regular workflow).

```python
second_half_indices = list(range(11, 21))
first_half_indices = [21-i for i in second_half_indices]
base_dataset = PalindromeDataset(size=1000, max_value=30, half_length=10).to(device)

# Get a set of palindromic tokens & non-palindromic tokens (with the second half of both tok sequences the same)
palindromic_tokens = base_dataset.toks.clone()
palindromic_tokens[:, 1:11] = palindromic_tokens[:, 11:21].flip(-1)
nonpalindromic_tokens = palindromic_tokens.clone()
# Use some modular arithmetic to make sure the sequence I'm creating is fully non-palindromic
nonpalindromic_tokens[:, 1:11] += t.randint_like(nonpalindromic_tokens[:, 1:11], low=1, high=30)
nonpalindromic_tokens[:, 1:11] %= 31

# Run with cache, and get attention differences
_, cache_palindromic = model.run_with_cache(palindromic_tokens, return_type=None)
_, cache_nonpalindromic = model.run_with_cache(nonpalindromic_tokens, return_type=None)
p1 = cache_palindromic["pattern", 0][:, 0, second_half_indices, second_half_indices] # [batch seqQ]
p1_prime = cache_palindromic["pattern", 0][:, 0, second_half_indices, first_half_indices] # [batch seqQ]
p2 = cache_nonpalindromic["pattern", 0][:, 0, second_half_indices, second_half_indices] # [batch seqQ]
p2_prime = cache_nonpalindromic["pattern", 0][:, 0, second_half_indices, first_half_indices] # [batch seqQ]

fig_names = ["fig6a", "fig6b"]

for diff, title in zip([p2 - p1, p1_prime - p2_prime], ["p<sub>2</sub> - p<sub>1</sub>", "p<sub>1</sub>' - p<sub>2</sub>'"]):
    fig = go.Figure(
        data = [
            go.Box(y=utils.to_numpy(diff[:, i]), name=f"({j1}, {j2})", boxpoints='suspectedoutliers')
            for i, (j1, j2) in enumerate(zip(first_half_indices, second_half_indices))
        ],
        layout = go.Layout(
            title = f"Attn diff: {title}",
            template = "simple_white",
            width = 800,
        )
    ).add_hline(y=0, opacity=1.0, line_color="black", line_width=1)
    fig.show()
    print(f"Avg diff (over non-adversarial tokens) = {diff[:, [i for i in range(10) if i not in [17-11, 19-11]]].mean():.3f}")
```
""", unsafe_allow_html=True)
    
    fig6a = go.Figure(json.loads(open(palindromes_dir / "fig6a.json", 'r').read()))
    fig6b = go.Figure(json.loads(open(palindromes_dir / "fig6b.json", 'r').read()))
    st.plotly_chart(fig6a, use_container_width=False)
    st.markdown(r"""<div style='font-family:monospace; font-size:15px;'>Avg diff (over non-adversarial tokens) = 0.373</div><br>""", unsafe_allow_html=True)
    st.plotly_chart(fig6b, use_container_width=False)
    st.markdown(r"""<div style='font-family:monospace; font-size:15px;'>Avg diff (over non-adversarial tokens) = 0.544</div><br>""", unsafe_allow_html=True)
    
    st.markdown(r"""
## Conclusion

Yep, it looks like this "attn diff" does generally separate palindromic and non-palindromic tokens very well. Also, remember that in most non-palindromic sequences there will be more than one non-palindromic token, so we don't actually need perfect separation most of the time. We'll use the conservative figure of $0.373$ as our coefficient when we perform logit attribution later.

A quick sidenote - when we add back in adversarial positions 17 & 19, the points are no longer cleanly separate. We can verify that in head `1.0`, the `END` token never attends to positions 17 & 19 (which makes sense, if these tokens don't contain useful information). Code showing this is below.

```python
layer_1_head = 0

tokens_are_palindromic = (dataset.toks == dataset.toks.flip(-1)) # (batch, seq)
attn = cache["pattern", 1][:, layer_1_head, -1] # (batch, src token)

attn_palindromes = [attn[tokens_are_palindromic[:, i], i].mean().item() for i in range(attn.shape[1])]
attn_nonpalindromes = [attn[~tokens_are_palindromic[:, i], i].mean().item() for i in range(attn.shape[1])]

bar(
    [attn_palindromes, attn_nonpalindromes], 
    names=["Token is palindromic", "Token is non-palindromic"],
    barmode="group",
    width=800,
    title=f"Average attention from END to other tokens, in head 1.{layer_1_head}",
    labels={"index": "Source position", "variable": "Token type", "value": "Attn"}, 
    template="simple_white",
    x=["START"] + list(map(str, range(1, 21))) + ["END"],
    xaxis_tickangle=-45,
)
```
""", unsafe_allow_html=True)
    
    fig7 = go.Figure(json.loads(open(palindromes_dir / "fig7.json", 'r').read()))
    st.plotly_chart(fig7, use_container_width=False)

    st.markdown(r"""
Another thing which this plot makes obvious is that position 20 is rarely attended to by head 1.0 (explaining the third advex we found above). However, if you look at the attention patterns for head 1.1, you can see that it picks up the slack by attending to position 20 a lot, especially for non-palindromes.

## Next steps

We want to try and formalize this composition between head 0.0 and 1.0. We think that K-composition (and possibly V-composition) is going on.

**Question - do you think this is more likely to involve positional information or token information?**

<details>
<summary>Answer</summary>

It's more likely to involve positional information.

From the model's perspective, if $x$ and $x'$ are different tokens, it doesn't matter if they're $24, 25$ or $25, 24$ - it's all the same. But the positional information which gets moved from $x' \to x$ will always be the same for each $x$, and same for the information which gets moved from $x \to x$. So it's more likely that the model is using that.

This means we should replace our quantity $(x - x')^T W_{OV}^{0.0}$ with $(p - p')^T W_{OV}^{0.0}$ (where $p$ is the positional vector). 

When it comes to layernorm, we can take the mean of the scale factors over the batch dimension, but preserve the seq dimension. We'll denote $\hat{p}$ and $\hat{p}'$ as the positional vectors after applying layernorm. Then we'll call $v_i = (\hat{p}_i - \hat{p}'_i)^T W_{OV}^{0.0}$ the "difference vector" for the $i$th token (where $i$ is a sequence position in the second half).

</details>

Let's use this to set up an experiment. We want to take this "difference vector" $v_i$, and show that (at least for the non-adversarial token positions $i$), this vector is associated with:

* Increasing the attention from `END` to itself (i.e. K-composition)
* Pushing for the "unbalanced" prediction when it's attended to (i.e. V-composition)

## First experiment: K-composition

For each of these difference vectors, we can compute the corresponding keys for head 1.0, and we can also get the query vectors from the `END` token and measure their cosine similarity. For the non-adversarial tokens, we expect a very high cosine similarity, indicating that the model has learned to attend from the `END` token back to any non-palindromic token in the second half.

There are advantages and disadvantages of using cosine similarity. The main disadvantage is that it doesn't tell you anything about magnitudes. The main advantage is that, by normalizing for scale, the information you get from it is more immediately interpretable (because you can use baselines such as "all cosine sims are between 0 and 1" and "the expected value of the cosine sim of two random vectors in N-dimensional space is zero, with a standard deviation of $\sqrt{1/N}$").

```python
def get_keys_and_queries(layer_1_head: int):

    scale0 = cache["scale", 0, "ln1"][:, :, 0].mean(0) # [seq 1]
    W_pos_scaled = model.W_pos / cache["scale", 0, "ln1"][:, :, 0].mean(0) # [seq d_model]

    W_pos_diff_vectors = W_pos_scaled[second_half_indices] - W_pos_scaled[first_half_indices] # [half_seq d_model]
    difference_vectors = W_pos_diff_vectors @ model.W_V[0, 0] @ model.W_O[0, 0] # [half_seq d_model]

    scale1 = cache["scale", 1, "ln1"][:, second_half_indices, layer_1_head].mean(0) # [half_seq 1]
    difference_vectors_scaled = difference_vectors / scale1 # [half_seq d_model]
    all_keys = difference_vectors_scaled @ model.W_K[1, layer_1_head] # [half_seq d_head]

    # Averaging queries over batch dimension (to make sure we're not missing any bias terms)
    END_query = cache["q", 1][:, -1, layer_1_head].mean(0) # [d_head]

    # Get the cosine similarity
    all_keys_normed = all_keys / all_keys.norm(dim=-1, keepdim=True)
    END_query_normed = END_query / END_query.norm()
    cos_sim = all_keys_normed @ END_query_normed

    assert cos_sim.shape == (10,)
    return cos_sim


cos_sim_L1H0 = get_keys_and_queries(0)
cos_sim_L1H1 = get_keys_and_queries(1)

imshow(
    t.stack([cos_sim_L1H0, cos_sim_L1H1]),
    title = "Cosine similarity between difference vector keys and END query",
    width = 850,
    height = 400,
    x = [f"({i}, {j})" for i, j in zip(first_half_indices, second_half_indices)],
    y = ["1.0", "1.1"],
    labels = {"x": "Token pair", "y": "Head"},
    text_auto = ".2f",
)
```
""", unsafe_allow_html=True)
    
    fig8 = go.Figure(json.loads(open(palindromes_dir / "fig8.json", 'r').read()))
    st.plotly_chart(fig8, use_container_width=False)

    st.markdown(r"""
## Conclusion

These results are very striking. We make the following conclusions:

* As expected, for most tokens in the second half, head 1.0 will attend more to any token which attended to itself in head 0.0.
* The exceptions are 17 & 19 (the adversarial tokens we observed earlier) and 20 (which we saw was a blind spot of head 1.0 when we looked at attention patterns earlier).
* Head 1.0 tries to compensate for the blind spots at these sequence positions, it does a particularly good job at position 20.

## Second experiment: V-composition

Let's look at the direct logit attribution we get when we feed this difference vector through the OV matrix of heads in layer 1. We can re-use a lot of our code from the previous function.

```python
def get_DLA(layer_1_head: int):

    W_pos_scaled = model.W_pos / cache["scale", 0, "ln1"][:, :, 0].mean(0)

    W_pos_diff_vectors = W_pos_scaled[second_half_indices] - W_pos_scaled[first_half_indices] # [half_seq d_model]
    difference_vectors = W_pos_diff_vectors @ model.W_V[0, 0] @ model.W_O[0, 0] # [half_seq d_model]

    # This is the average multiple of this vector that gets added to the non-palindromic tokens relative to the
    # palindromic tokens (from the experiment we ran earlier)
    difference_vectors *= 0.373

    scale1 = cache["scale", 1, "ln1"][:, second_half_indices, layer_1_head].mean(0) # [half_seq 1]
    difference_vectors_scaled = difference_vectors / scale1
    all_outputs = difference_vectors_scaled @ model.W_V[1, layer_1_head] @ model.W_O[1, layer_1_head]

    # Scale & get direct logit attribution
    final_ln_scale = cache["scale"][~dataset.is_palindrome.bool(), -1].mean()
    all_outputs_scaled = all_outputs / final_ln_scale
    logit_attribution = all_outputs_scaled @ model.W_U
    # Get logit diff (which is positive for the "non-palindrome" classification)
    logit_diff = logit_attribution[:, 0] - logit_attribution[:, 1]

    return logit_diff


dla_L1H0 = get_DLA(0)
dla_L1H1 = get_DLA(1)
dla_L1 = t.stack([dla_L1H0, dla_L1H1])

imshow(
    dla_L1,
    title = "Direct logit attribution for the path W<sub>pos</sub> 'difference vectors' ➔ 0.0 ➔ (1.0 & 1.1) ➔ logits",
    width = 850,
    height = 400,
    x = [f"({i}, {j})" for i, j in zip(first_half_indices, second_half_indices)],
    y = ["1.0", "1.1"],
    labels = {"x": "Token pair", "y": "Head"},
    text_auto = ".2f",
)
```
""", unsafe_allow_html=True)
    
    fig9 = go.Figure(json.loads(open(palindromes_dir / "fig9.json", 'r').read()))
    st.plotly_chart(fig9, use_container_width=False)

    st.markdown(r"""
## Conclusions

* The results for head 1.0 agree with our expectation. The values in the 3 adversarial cases don't matter because `END` never pays attention to these tokens.
* The results for head 1.1 show us that this head compensates for the blind spot at position 20, but not at positions 17 or 19.
* The sizes of DLAs look about reasonable - in particular, the size of DLA for head 1.0 on all the non-adversarial positions is only a bit larger than the empirically observed logit diff (which is about 7.5 - see code cell below), which makes sense given that head 1.0 will usually pay very large (but not quite 100%) attention to non-palindromic tokens in the second half of the sequence, conditional on some non-palindromic tokens existing.

<br>

# A few more experiments

I consider the main problem to have basically been solved now, but here are a few more experiments we can run that shed more light on the model.

## Targeted ablations

Our previous results suggested that both 0.1 and 1.1 seem to compensate for blind spots at position 20. We should guess that mean ablating them everywhere except at position 20 shouldn't change the loss by much at all.

In the case of head 0.1, we should mean ablate the result everywhere except position 20 (because it's the output at this position that we care about). In the case of head 1.1, we should mean ablate the value vectors everywhere except position 20 (because it's the input at this position that we care about).

Note - in this case we're measuring loss rather than logit diff. This is because the purpose of heads 0.1 and 1.1 is to fix the model's blind spots, not to increase logit diff overall. It's entirely possible for a head to decrease loss and increase logit diff (in fact this is what we see for head 1.1).

```python
def targeted_mean_ablation_loss(
    head: Tuple[int, int],
    ablation_type: Literal["input", "output"],
    ablate_20: bool
):

    # Get values for doing mean ablation everywhere (except possibly position 20)
    layer, head_idx = head
    component = "result" if ablation_type == "output" else "v"
    seq_pos_to_ablate = slice(None) if ablate_20 else [i for i in range(22) if i != 20]
    ablation_values = cache[component, layer][:, seq_pos_to_ablate, head_idx].mean(0) # [seq d_model]

    # Define hook function
    def hook_patch_mean(activation: Float[Tensor, "batch seq nheads d"], hook: HookPoint):
        activation[:, seq_pos_to_ablate, head_idx] = ablation_values
        return activation

    # Run hooked forward pass
    model.reset_hooks()
    logits = model.run_with_hooks(
        dataset.toks,
        fwd_hooks = [(utils.get_act_name(component, layer), hook_patch_mean)],
    )
    logprobs = logits[:, -1].log_softmax(-1)
    logprobs_correct = t.where(dataset.is_palindrome.bool(), logprobs[:, 1], logprobs[:, 0])
    return -logprobs_correct.mean().item()


print(f"Original loss                           = {avg_cross_entropy_loss:.3f}\n")
print(f"0.1 ablated everywhere (incl. posn 20)  = {targeted_mean_ablation_loss((0, 1), 'output', ablate_20=True):.3f}")
print(f"0.1 ablated everywhere (except posn 20) = {targeted_mean_ablation_loss((0, 1), 'output', ablate_20=False):.3f}\n")
print(f"1.1 ablated everywhere (incl. posn 20)  = {targeted_mean_ablation_loss((1, 1), 'input', ablate_20=True):.3f}")
print(f"1.1 ablated everywhere (except posn 20) = {targeted_mean_ablation_loss((1, 1), 'input', ablate_20=False):.3f}")
```

<div style='font-family:monospace; font-size:15px;'>
Original loss                           = 0.008<br>
<br>
0.1 ablated everywhere (incl. posn 20)  = 0.118<br>
0.1 ablated everywhere (except posn 20) = 0.010<br>
<br>
1.1 ablated everywhere (incl. posn 20)  = 0.014<br>
1.1 ablated everywhere (except posn 20) = 0.008
</div><br>

## Composition scores

You can also measure composition scores (see the fourth section of [Intro to Mech Interp](https://arena-ch1-transformers.streamlit.app/[1.2]_Intro_to_Mech_Interp) for more details on what these are). Also, see Andy Arditi's solutions for an implementation of composition scores for this problem. These plots demonstrate strong composition between heads 0.0 and 1.0, and much weaker for all other heads (which is what we expect, since the other heads only compose in a narrow range of situations).

## How is the "non-palindromic" information stored?

We can look at the cosine similarity between the "difference vectors" for each sequence position (code below). The result - cosine similarity is extremely high for all tokens except for the advex positions 17, 19 and 20. This implies that (for these non-advex token positions), the information getting stored in each sequence position in the second half is boolean - i.e. there is a well-defined direction in residual stream space which represents "this token is not palindromic", and this direction is the same for all non-advex positions in the second half of the sequence.

It makes sense that this result doesn't hold for 17 and 19 (because 0.0's attention doesn't work for these positions, so there's no signal that can come from here). Interestingly, the fact that this result doesn't hold for 20 reframes the question of why 20 is adversarial - it's not because it's a blind spot of head 1.0, it's because it's a blind spot of the QK circuit of head 0.0.

```python
W_pos_scaled = model.W_pos / cache["scale", 0, "ln1"][:, :, 0].mean(0)

W_pos_difference_vectors = W_pos_scaled[second_half_indices] - W_pos_scaled[first_half_indices]
difference_vectors = W_pos_difference_vectors @ model.W_V[0, 0] @ model.W_O[0, 0]

difference_vectors_normed = difference_vectors / difference_vectors.norm(dim=-1, keepdim=True)

cos_sim = difference_vectors_normed @ difference_vectors_normed.T

imshow(
    cos_sim,
    x = [f"({i}, {j})" for i, j in zip(first_half_indices, second_half_indices)],
    y = [f"({i}, {j})" for i, j in zip(first_half_indices, second_half_indices)],
    title = "Cosine similarity of 'difference vectors' at different positions",
    width = 700,
    height = 600,
    text_auto = ".2f",
)
```
""", unsafe_allow_html=True)
    
    fig10 = go.Figure(json.loads(open(palindromes_dir / "fig10.json", 'r').read()))
    st.plotly_chart(fig10, use_container_width=False)


def section_0_august():

    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li><a class='contents-el' href='#prerequisites'>Prerequisites</a></li>
    <li><a class='contents-el' href='#difficulty'>Difficulty</a></li>
    <li><a class='contents-el' href='#motivation'>Motivation</a></li>
    <li><a class='contents-el' href='#logistics'>Logistics</a></li>
    <li><a class='contents-el' href='#what-counts-as-a-solution'>What counts as a solution?</a></li>
    <li><a class='contents-el' href='#setup'>Setup</a></li>
    <li><a class='contents-el' href='#task-dataset'>Task & Dataset</a></li>
    <li><a class='contents-el' href='#model'>Model</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(
r"""
# Monthly Algorithmic Challenge (August 2023): First Unique Character

### Colab: [problem](https://colab.research.google.com/drive/15huO8t1io2oYuLdszyjhMhrPF3WiWhf1) | [solutions](https://colab.research.google.com/drive/1E22t3DP5F_MEDNepARlrZy-5w7bv0_8G)

This post is the second in the sequence of monthly mechanistic interpretability challenges. They are designed in the spirit of [Stephen Casper's challenges](https://www.lesswrong.com/posts/KSHqLzQscwJnv44T8/eis-vii-a-challenge-for-mechanists), but with the more specific aim of working well in the context of the rest of the ARENA material, and helping people put into practice all the things they've learned so far.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/writer.png" width="350">

## Prerequisites

The following ARENA material should be considered essential:

* **[1.1] Transformer from scratch** (sections 1-3)
* **[1.2] Intro to Mech Interp** (sections 1-3)

The following material isn't essential, but is recommended:

* **[1.2] Intro to Mech Interp** (section 4)
* **July's Algorithmic Challenge - writeup** (on the sidebar of this page)

## Difficulty

This problem is a step up in difficulty to the July problem. The algorithmic problem is of a similar flavour, and the model architecture is very similar (the main difference is that this model has 3 attention heads per layer, instead of 2).

## Motivation

Neel Nanda's post [200 COP in MI: Interpreting Algorithmic Problems](https://www.lesswrong.com/posts/ejtFsvyhRkMofKAFy/200-cop-in-mi-interpreting-algorithmic-problems) does a good job explaining the motivation behind solving algorithmic problems such as these. I'd strongly recommend reading the whole post, because it also gives some high-level advice for approaching such problems.

The main purpose of these challenges isn't to break new ground in mech interp, rather they're designed to help you practice using & develop better understanding for standard MI tools (e.g. interpreting attention, direct logit attribution), and more generally working with libraries like TransformerLens.

Also, they're hopefully pretty fun, because why shouldn't we have some fun while we're learning?

## Logistics

The solution to this problem will be published on this page at the start of October, at the same time as the next problem in the sequence. There will also be an associated LessWrong post.

If you try to interpret this model, you can send your attempt in any of the following formats:

* Colab notebook,
* GitHub repo (e.g. with ipynb or markdown file explaining results),
* Google Doc (with screenshots and explanations),
* or any other sensible format.

You can send your attempt to me (Callum McDougall) via any of the following methods:

* The [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), via a direct message to me
* My personal email: `cal.s.mcdougall@gmail.com`
* LessWrong message ([here](https://www.lesswrong.com/users/themcdouglas) is my user)

**I'll feature the names of everyone who sends me a solution on this website, and also give a shout out to the best solutions.**

Please don't discuss specific things you've found about this model until the challenge is over (although you can discuss general strategies and techniques, and you're also welcome to work in a group if you'd like). The deadline for this problem will be the end of this month, i.e. 31st August.

## What counts as a solution?

Going through the solutions for the previous problem in the sequence (July: Palindromes) as well as the exercises in **[1.4] Balanced Bracket Classifier** should give you a good idea of what I'm looking for. In particular, I'd expect you to:

* Describe a mechanism for how the model solves the task, in the form of the QK and OV circuits of various attention heads (and possibly any other mechanisms the model uses, e.g. the direct path, or nonlinear effects from layernorm),
* Provide evidence for your mechanism, e.g. with tools like attention plots, targeted ablation / patching, or direct logit attribution.
* (Optional) Include additional detail, e.g. identifying the subspaces that the model uses for certain forms of information transmission, or using your understanding of the model's behaviour to construct adversarial examples.

# Setup

```python
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from pathlib import Path

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "monthly_algorithmic_problems" / "august23_unique_char"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from monthly_algorithmic_problems.august23_unique_char.dataset import UniqueCharDataset
from monthly_algorithmic_problems.august23_unique_char.model import create_model
from plotly_utils import hist, bar, imshow

device = t.device("cuda" if t.cuda.is_available() else "cpu")
```

## Task & Dataset

The algorithmic task is as follows: the model is presented with a sequence of characters, and for each character it has to correctly identify the first character in the sequence (up to and including the current character) which is unique up to that point.

The null character `"?"` has two purposes:

* In the input, it's used as the start character (because it's often helpful for interp to have a constant start character, to act as a "rest position").
* In the output, it's also used as the start character, **and** to represent the classification "no unique character exists".

Here is an example of what this dataset looks like:

```python
dataset = UniqueCharDataset(size=2, vocab=list("abc"), seq_len=6, seed=42)

for seq, first_unique_char_seq in zip(dataset.str_toks, dataset.str_tok_labels):
    print(f"Seq = {''.join(seq)}, Target = {''.join(first_unique_char_seq)}")
```

<div style='font-family:monospace; font-size:15px;'>
Seq = ?acbba, Target = ?aaaac<br>
Seq = ?cbcbc, Target = ?ccb??
</div><br>

Explanation:

1. In the first sequence, `"a"` is unique in the prefix substring `"acbb"`, but it repeats at the 5th sequence position, meaning the final target character is `"c"` (which appears second in the sequence).
2. In the second sequence, `"c"` is unique in the prefix substring `"cb"`, then it repeats so `"b"` is the new first unique token, and for the last 2 positions there are no unique characters (since both `"b"` and `"c"` have been repeated) so the correct classification is `"?"` (the "null character").

The relevant files can be found at:

```
chapter1_transformers/
└── exercises/
    └── monthly_algorithmic_problems/
        └── august23_unique_char/
            ├── model.py               # code to create the model
            ├── dataset.py             # code to define the dataset
            ├── training.py            # code to training the model
            └── training_model.ipynb   # actual training script
```

We've given you the class `UniqueCharDataset` to store your data, as you can see above. You can slice this object to get batches of tokens and labels (e.g. `dataset[:5]` returns a length-2 tuple, containing the 2D tensors representing the tokens and correct labels respectively). You can also use `dataset.toks` or `dataset.labels` to access these tensors directly, or `dataset.str_toks` and `dataset.str_tok_labels` to get the string representations of the tokens and labels (like we did in the code above).

## Model

Our model was trained by minimising cross-entropy loss between its predictions and the true labels, at every sequence position simultaneously (including the zeroth sequence position, which is trivial because the input and target are both always `"?"`). You can inspect the notebook `training_model.ipynb` to see how it was trained. I used the version of the model which achieved highest accuracy over 50 epochs (accuracy ~99%).

The model is is a 2-layer transformer with 3 attention heads, and causal attention. It includes layernorm, but no MLP layers. You can load it in as follows:

```python
filename = section_dir / "first_unique_char_model.pt"

model = create_model(
    seq_len=20,
    vocab=list("abcdefghij"),
    seed=42,
    d_model=42,
    d_head=14,
    n_layers=2,
    n_heads=3,
    normalization_type="LN",
    d_mlp=None # attn-only model
)

state_dict = t.load(filename)

state_dict = model.center_writing_weights(t.load(filename))
state_dict = model.center_unembed(state_dict)
state_dict = model.fold_layer_norm(state_dict)
state_dict = model.fold_value_biases(state_dict)
model.load_state_dict(state_dict, strict=False);
```

The code to process the state dictionary is a bit messy, but it's necessary to make sure the model is easy to work with. For instance, if you inspect the model's parameters, you'll see that `model.ln_final.w` is a vector of 1s, and `model.ln_final.b` is a vector of 0s (because the weight and bias have been folded into the unembedding).

```python
print("ln_final weight: ", model.ln_final.w)
print("\nln_final, bias: ", model.ln_final.b)
```

<details>
<summary>Aside - the other weight processing parameters</summary>

Here's some more code to verify that our weights processing worked, in other words:

* The unembedding matrix has mean zero over both its input dimension (`d_model`) and output dimension (`d_vocab`)
* All writing weights (i.e. `b_O`, `W_O`, and both embeddings) have mean zero over their output dimension (`d_model`)
* The value biases `b_V` are zero (because these can just be folded into the output biases `b_O`)

```python
W_U_mean_over_input = einops.reduce(model.W_U, "d_model d_vocab -> d_model", "mean")
t.testing.assert_close(W_U_mean_over_input, t.zeros_like(W_U_mean_over_input))

W_U_mean_over_output = einops.reduce(model.W_U, "d_model d_vocab -> d_vocab", "mean")
t.testing.assert_close(W_U_mean_over_output, t.zeros_like(W_U_mean_over_output))

W_O_mean_over_output = einops.reduce(model.W_O, "layer head d_head d_model -> layer head d_head", "mean")
t.testing.assert_close(W_O_mean_over_output, t.zeros_like(W_O_mean_over_output))

b_O_mean_over_output = einops.reduce(model.b_O, "layer d_model -> layer", "mean")
t.testing.assert_close(b_O_mean_over_output, t.zeros_like(b_O_mean_over_output))

W_E_mean_over_output = einops.reduce(model.W_E, "token d_model -> token", "mean")
t.testing.assert_close(W_E_mean_over_output, t.zeros_like(W_E_mean_over_output))

W_pos_mean_over_output = einops.reduce(model.W_pos, "position d_model -> position", "mean")
t.testing.assert_close(W_pos_mean_over_output, t.zeros_like(W_pos_mean_over_output))

b_V = model.b_V
t.testing.assert_close(b_V, t.zeros_like(b_V))
```

</details>

The model's output is a logit tensor, of shape `(batch_size, seq_len, d_vocab+1)`. The `[i, j, :]`-th element of this tensor is the logit distribution for the label at position `j` in the `i`-th sequence in the batch. The first `d_vocab` elements of this tensor correspond to the elements in the vocabulary, and the last element corresponds to the null character `"?"` (which is not in the input vocab).

A demonstration of the model working:

```python
dataset = UniqueCharDataset(size=1000, vocab=list("abcdefghij"), seq_len=20, seed=42)

logits, cache = model.run_with_cache(dataset.toks)

logprobs = logits.log_softmax(-1) # [batch seq_len d_vocab]
probs = logprobs.softmax(-1) # [batch seq_len d_vocab]

batch_size, seq_len = dataset.toks.shape
logprobs_correct = logprobs[t.arange(batch_size)[:, None], t.arange(seq_len)[None, :], dataset.labels] # [batch seq_len]
probs_correct = probs[t.arange(batch_size)[:, None], t.arange(seq_len)[None, :], dataset.labels] # [batch seq_len]

avg_cross_entropy_loss = -logprobs_correct.mean().item()
avg_correct_prob = probs_correct.mean().item()
min_correct_prob = probs_correct.min().item()

print(f"Average cross entropy loss: {avg_cross_entropy_loss:.3f}")
print(f"Average probability on correct label: {avg_correct_prob:.3f}")
print(f"Min probability on correct label: {min_correct_prob:.3f}")
```

<div style='font-family:monospace; font-size:15px;'>
Average cross entropy loss: 0.017<br>
Average probability on correct label: 0.988<br>
Min probability on correct label: 0.001
</div><br>

And a visualisation of its probability output for a single sequence:

```python
def show(i):
    imshow(
        probs[i].T,
        y=dataset.vocab,
        x=[f"{dataset.str_toks[i][j]}<br><sub>({j})</sub>" for j in range(model.cfg.n_ctx)],
        labels={"x": "Token", "y": "Vocab"},
        xaxis_tickangle=0,
        title=f"Sample model probabilities (for batch idx = {i}), with correct classification highlighted",
        text=[
            ["〇" if str_tok == correct_str_tok else "" for correct_str_tok in dataset.str_tok_labels[i]]
            for str_tok in dataset.vocab
        ],
        width=900,
        height=450,
    )

show(0)
```
""", unsafe_allow_html=True)
    
    with open(unique_char_dir / "fig_demo.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=500)

    st.markdown(r"""
If you want some guidance on how to get started, I'd recommend reading the solutions for the July & August problems - I expect there to be a lot of overlap in the best way to tackle these two problems. You can also reuse some of that code!

Best of luck! 🎈

""", unsafe_allow_html=True)


def section_1_august():
    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li><a class='contents-el' href='#summary-of-how-the-model-works'>Summary of how the model works</a></li>
    <li><a class='contents-el' href='#some-initial-notes'>Some initial notes</a></li>
    <li><a class='contents-el' href='#attention-patterns'>Attention patterns</a></li>
    <li><a class='contents-el' href='#ov-circuits'>OV circuits</a></li>
    <li><a class='contents-el' href='#qk-circuits'>QK circuits</a></li>
    <li><a class='contents-el' href='#direct-logit-attribution'>Direct Logit Attribution</a></li>
    <li><a class='contents-el' href='#final-summary'>Final summary</a></li>
    <li><a class='contents-el' href='#adversarial-examples'>Adversarial examples</a></li>
    <li><a class='contents-el' href='#remaining-questions-notes-things-not-discussed'>Remaining questions / notes / things not discussed</a></li>
</ul></li>""", unsafe_allow_html=True)
    
    st.markdown(
r"""
# Monthly Algorithmic Challenge (August 2023): Solutions

We assume you've run all the setup code from the previous page "[August] First Unique Token". Here's all the new setup code you'll need:
                
```python
%pip install git+https://github.com/callummcdougall/eindex.git

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import json
from typing import List, Union, Optional, Callable, cast
import torch as t
from torch import Tensor
import plotly.express as px
import einops
from jaxtyping import Float, Int, Bool
from pathlib import Path
import pandas as pd
import circuitsvis as cv
from IPython.display import display
from transformer_lens import ActivationCache
from eindex import eindex

t.set_grad_enabled(False)

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "monthly_algorithmic_problems" / "august23_unique_char"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from monthly_algorithmic_problems.august23_unique_char.model import create_model
from monthly_algorithmic_problems.august23_unique_char.dataset import UniqueCharDataset, find_first_unique
from plotly_utils import imshow

device = t.device("cuda" if t.cuda.is_available() else "cpu")

dataset = UniqueCharDataset(size=1000, vocab=list("abcdefghij"), seq_len=20, seed=42)
```

And for getting some activations:

```python
logits, cache = model.run_with_cache(dataset.toks)
logits = cast(Tensor, logits)

logprobs = logits.log_softmax(-1) # [batch seq_len vocab_out]
probs = logprobs.softmax(-1)

batch_size, seq_len = dataset.toks.shape
logprobs_correct = eindex(logprobs, dataset.labels, "batch seq [batch seq]")
probs_correct = eindex(probs, dataset.labels, "batch seq [batch seq]")

avg_cross_entropy_loss = -logprobs_correct.mean().item()
avg_correct_prob = probs_correct.mean().item()
min_correct_prob = probs_correct.min().item()

print(f"\nAverage cross entropy loss: {avg_cross_entropy_loss:.3f}")
print(f"Average probability on correct label: {avg_correct_prob:.3f}")
print(f"Min probability on correct label: {min_correct_prob:.3f}")
```

Output:

<div style='font-family:monospace; font-size:15px;'>
Average cross entropy loss: 0.017<br>
Average probability on correct label: 0.988<br>
Min probability on correct label: 0.001
</div><br>

# Summary of how the model works

In case you don't want to read the entire solution, I'll present a summary of it here (glossing over the actual research process I went through). You can read approximately the same thing in the [LessWrong post](https://www.lesswrong.com/posts/67xQqsimxywp9wYjC/mech-interp-challenge-september-deciphering-the-addition).

The key idea with this model is **path decomposition** (see the [corresponding section](https://transformer-circuits.pub/2021/framework/index.html#three-kinds-of-composition) of A Mathematical Framework for Transformer Circuits). There are several different important types of path in this model, with different interpretations & purposes. We might call these **negative paths** and **positive paths**. The negative paths are designed to suppress repeated tokens, and the positive paths are designed to boost tokens which are more likely to be the first unique token.

Let's start with the **negative paths**. Some layer 0 heads are **duplicate token heads**; they're composing with layer 1 heads to cause those heads to attend to & suppress duplicated tokens. This is done both with K-composition (heads in layer 1 suppress duplicated tokens because they attend to them more), and V-composition (the actual outputs of the DTHs are used as value input to heads in layer 1 to suppress duplicated tokens). Below is an example, where the second and third instances of a attend back to the first instance of a in **head 0.2**, and this composes with **head 1.0** which attends back to (and suppresses) all the duplicated a tokens.

<img src="https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/67xQqsimxywp9wYjC/fkeofkykojp22nlxwk4y" width="700">
<img src="https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/67xQqsimxywp9wYjC/fqqbgblysh4mipye7s1e" width="700">

Now, let's move on to the **positive paths**. Heads in layer 0 will attend to early tokens which aren't the same as the current destination token, because both these bits of evidence correlate with this token being the first unique token at this position (this is most obvious with the second token, since the first token is the correct answer here if and only if it doesn't equal the second token). Additionally, the outputs of heads in layer 0 are used as **value input** to heads in layer 1 to boost these tokens, i.e. as a virtual OV circuit. These paths aren't as obviously visible in the attention probabilities, because they're distributed: many tokens will weakly attend to some early token in a layer-0 head, and then all of those tokens will be weakly attended to by some layer-1 head. But the paths can be seen when we plot all the OV circuits, coloring each value by how much the final logits for that token are affected at the destination position:

<img src="https://res.cloudinary.com/lesswrong-2-0/image/upload/f_auto,q_auto/v1/mirroredImages/67xQqsimxywp9wYjC/zmerzkwzh05r9bs0grm2" width="700">

Another important thing to note - these paths aren't just split by head, they're also split by character. For example, the path with heads 0.2 & 1.0 is a negative path for token `a` (we saw evidence of this in the attention patterns earlier). But it's a positive path for token `c`, and indeed when we look at the dataset we can see evidence of head 0.2 attending to early instances of the `c` token, and this being used by head 1.0 to boost `c`. Also, note how all letters in the vocabulary are covered by at least one head: the paths through head 1.0 manage boosting / suppression for tokens `[a, c]`, and heads 1.1, 1.2 manage `[d, e, f, j]` and `[b, h, i]` respectively. The disjoint union of these groups is the whole vocabulary.

# Some initial notes

I initially expected the same high-level story as July's model:

* There are some layer-0 heads which are moving information into useful sequence positions, depending on whether the tokens at those sequence positions are the same as earlier tokens.
* There are some layer-1 heads which are picking up on this information, and converting it into a useful classification.

Things I expect to see:

* Attention patterns
    * There are layer-0 heads which act as duplicate token heads (abbrev. DTH); attending back to previous instances of themselves.
    * There are layer-1 heads for which each token attends back to the first unique token up to that point.
* Full matrices
    * The full QK matrix of heads in layer 0 should be essentially the identity, if they're acting as duplicate token heads.
    * The OV circuit of heads in layer 1 should basically be a copying circuit, because when they attend to token `T` they're using that as a prediction. 
        * The OV circuit could be V-composing with layer-0 heads, but this doesn't seem strictly necessary.
    * The full QK matrix of heads in layer 1 should be privileging earlier tokens (because if there's more than one unique token, the head will have to attend to the earlier one).
    * The full QK matrix of heads in layer 1 (with Q-composition from layer 0) should have a negative stripe, because they'll be avoiding tokens which are duplicates.

Other thoughts:

* With multiple low-dimensional heads, it's possible their functionality is being split. We kinda saw this in the July problem (with posn 20 being mostly handled by head 1.1 and the other positions in the second half being handled by head 1.0 to varying degrees of success). 

# Attention patterns

I've visualised attention patterns below. I wrote a function to perform some HTML formatting (aided by ChatGPT) to make the sequences easier to interpret, by highlighting all the tokens which are the first unique character *at some point in the sequence*. Also, note that `batch_labels` was supplied not as a list of strings, but as a function mapping (batch index, str toks) to a string. Either are accepted by the `cv.attention.from_cache` function.

```python
def format_sequence(str_toks: List[str], str_tok_labels: Tensor, code: bool = True) -> str:
    '''
    Given a sequence (as list of strings) and labels (as Int-type tensor), formats the sequence by
    highlighting all the tokens which are the first unique char at some point.
    
    We add an option to remove the code tag, because this doesn't work as a plotly title.
    '''
    seq = "<b><code>" + " ".join([
        f"<span style='color:red;'>{tok}</span>"
        if (tok in str_tok_labels) and (tok not in str_toks[:i]) else tok
        for i, tok in enumerate(str_toks)
    ]) + "</code></b>"
    if not(code):
        seq = seq.replace("<code>", "").replace("</code>", "")
    return seq

cv.attention.from_cache(
    cache = cache,
    tokens = dataset.str_toks,
    batch_idx = list(range(10)),
    attention_type = "info-weighted",
    radioitems = True,
    return_mode = "view",
    batch_labels = lambda batch_idx, str_tok_list: format_sequence(str_tok_list, dataset.str_tok_labels[batch_idx]),
    mode = "small",
)
```
""", unsafe_allow_html=True)
    
    with open(unique_char_dir / "fig_attn_1.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=620)

    st.markdown(
r"""
## Conclusions

Some of the evidence fits my model: 

* 0.1 is acting as a DTH, for all tokens except `a`
* 0.2 seems to be acting as a DTH for `a`, patching this gap
* I guessed some kind of split functionality might be happening in the layer-1 attention heads. The non-intersecting vertical lines in the layer-1 attention heads support this/
    * To clarify - the vertical lines suggest that each of these heads has a particular set of tokens which it cares about, and these sets are disjoint. Further examination shows that the sets seem to be `[a, c]` for 1.0, `[d, e, f, j]` for 1.1, and `[b, g, h, i]` for 1.2.
    * Notation: we'll refer to these sets as the "domain" of the layer 1 head. So `[a, c]` is the domain of 1.0, etc.

But a lot of the evidence doesn't fit:

* Head 0.0 attends pretty uniforly to early tokens which aren't same as the destination token - this doesn't fit with my DTH model
    * Guess: 0.0 is V-composing with heads in layer 1, because the combined heuristics "token is early" and "token is not the same as destination token" all seem like they point towards this token being a correct classification.
* Head 0.2 looks a bit like 0.0 in some respects - guess that it's doing some combination of DTH + whatever 0.0 is doing.
* Heads 0.0 and 0.2 are also strongly self-attending at the first non-null token. This fits with the guess above, because the first non-null token must be unique at that point (and is the best default guess for the first unique token later on in the sequence).
* Although layer 1 attention heads are splitting functionality in a way I thought might happen, they're not actually doing what I was expecting. I thought I'd see these heads attending to & boosting the first unique token, but they don't seem to be doing this.

From all this evidence, we form a new hypothesis:

> Some layer 0 heads (0.1 and partly 0.2) are DTH; they're K-composing with layer 1 heads to cause those heads to attend to & suppress duplicate tokens. Some layer 0 heads (0.0 and partly 0.2) are "early unique" heads; they're attending more to early tokens and are V-composing with layer 1 heads to boost these tokens. Also, layer 1 heads are splitting functionality across tokens in the vocabulary: each head in layer 1 has a particular set of tokens, whose disjoint union is the whole vocabulary.

This hypothesis turns out to be pretty much correct, minus a few details.

# OV circuits

The next thing I wanted to do was plot all OV circuits: both for the actual attention heads and the virtual attention heads. Before doing this, I wanted to make clear predictions about what I'd see based on the two different versions of my hypothesis:

* **Original hypothesis**
    * The OV circuits of layer 1 heads will be negative copying circuits (because they're attending to & suppressing duplicated tokens).
    * The virtual OV circuits from composition of heads in layers 0 & 1 will either not be important, or be negative copying circuits too.
* **New hypothesis**
    * The virtual OV circuits from 0.1 ➔ (head in layer 1) will be negative, specifically on that layer 1 head's domain. Same for the virtual circuit from 0.2 ➔ 1.0 for `a` (because `a` is in the domain of 1.0). 
    * The other virtual OV circuits will be positive copying circuits.
    * The OV circuits for layer 0 heads will probably be positive copying circuits for 0.0 and (0.2, minus the `a` token). Not sure what the OV circuits for layer 1 heads will look like, since the heads in layer 1 have to boost *and* suppress tokens (rather than just suppressing tokens, as in my original hypothesis). 
    * If the OV circuits for layer 0 are positive copying circuits, I'd weakly guess they'd be negative for layer 1.

```python
scale_final = cache["scale"][:, :, 0][:, 1:].mean()
scale_0 = cache["scale", 0, "ln1"][:, 1:].mean()
scale_1 = cache["scale", 1, "ln1"][:, 1:].mean()
W_OV = model.W_V @ model.W_O
W_E = model.W_E
W_U = model.W_U

# ! Get direct path
W_E_OV_direct = (W_E / scale_final) @ W_U

# ! Get full OV matrix for path through just layer 0
W_E_OV_0 = (W_E / scale_0) @ W_OV[0]
W_OV_0_full = (W_E_OV_0 / scale_final) @ W_U # [head1 vocab_in vocab_out]

# ! Get full OV matrix for path through just layer 1
W_E_OV_1 = (W_E / scale_1) @ W_OV[1]
W_OV_1_full = (W_E_OV_1 / scale_final) @ W_U # [head1 vocab_in vocab_out]

# ! Get full OV matrix for path through heads in layer 0 and 1
W_E_OV_01 = einops.einsum(
    (W_E_OV_0 / scale_1), W_OV[1],
    "head0 vocab_in d_model_in, head1 d_model_in d_model_out -> head0 head1 vocab_in d_model_out",
)
W_OV_01_full = (W_E_OV_01 / scale_final) @ W_U # [head0 head1 vocab_in vocab_out]

# Stick 'em together
assert W_OV_01_full.shape == (3, 3, 11, 11)
assert W_OV_1_full.shape == (3, 11, 11)
assert W_OV_0_full.shape == (3, 11, 11)
assert W_E_OV_direct.shape == (11, 11)
W_OV_full_all = t.cat([
    t.cat([W_E_OV_direct[None, None], W_OV_0_full[:, None]]), # [head0 1 vocab_in vocab_out]
    t.cat([W_OV_1_full[None], W_OV_01_full]),  # [head0 head1 vocab_in vocab_out]
], dim=1) # [head0 head1 vocab_in vocab_out]
assert W_OV_full_all.shape == (4, 4, 11, 11)

# Visually check plots are in correct order
# W_OV_full_all[0, 1] += 100
# W_OV_full_all[1, 0] += 100

components_0 = ["WE"] + [f"0.{i}" for i in range(3)]
components_1 = ["WU"] + [f"1.{i}" for i in range(3)]

# Text added after creating this plot, to highlight the stand-out patterns
text = []
patterns = ["", "", "", "", "ac", "ac", "c", "ac", "defj", "dj", "defj", "efj", "bgh", "bgi", "bhi", "bgh"]
for i, pattern in enumerate(patterns):
    text.append([[pattern[pattern.index(i)] if (i==j and i in pattern) else "" for i in dataset.vocab] for j in dataset.vocab])

imshow(
    W_OV_full_all.transpose(0, 1).flatten(0, 1), # .softmax(dim=-1),
    facet_col = 0,
    facet_col_wrap = 4,
    facet_labels = [" ➔ ".join(list(dict.fromkeys(["WE", c0, c1, "WU"]))) for c1 in components_1 for c0 in components_0],
    title = f"Full virtual OV circuits",
    x = dataset.vocab,
    y = dataset.vocab,
    labels = {"x": "Source", "y": "Dest"},
    height = 1200,
    width = 1200,
    text = text,
)
```
""", unsafe_allow_html=True)
    
    with open(unique_char_dir / "fig_virtual_ov.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=1100)

    st.markdown(
r"""
## Conclusion

These results basically fit with my new hypothesis, and I consider this plot and the conclusions drawn from it to be the central figure for explaining this model.

To review the ways in which this plot fits with my new hypothesis:

* The paths from 0.1 ➔ (head in layer 1) are mostly negative on that head's domain.
* The path from 0.2 ➔ 1.0 is negative at `a`.
* Most of the other paths from (0.0 or 0.2) ➔ (head in layer 1) are strongly positive on that layer 1 head's domain (and weakly positive outside of that head's domain).
* The OV circuits for heads 0.0 and 0.2 are positive copying circuits, albeit weakly so.
* We can see that the direct paths via a head in layer 1 are generally negative (more so than I expected).

Additionally, there is strong evidence against my original hypothesis: several of the virtual OV circuits are unmistakably positive copying circuits.

The ways in which this plot doesn't fit with my new hypothesis:

* There are two more negative paths from (0.0 ➔ 0.2) ➔ (head in layer 1) than I expected: both `b` and `g` have negative paths from 0.2 ➔ 1.2.
    * From looking at the rest of the graph, I'm guessing this is because the 0.1 ➔ 1.2 path doesn't do a very good job suppressing duplicated `b` or `g`, so another path has to step in and perform this suppression.

Some more notes on this visualisation:
* This plot might make it seem like the virtual paths are way more important than the single attention head paths. This is partly true, but also can be slightly misleading - these virtual paths will have smaller magnitudes than the plot suggests, since the attention patterns are formed by multiplying together two different attention patterns (and as we saw from the info-weighted attention patterns above, a lot of attention goes to the null character at the start of the sequence, and the result vector from this is very small so unlikely to be used in composition).

# QK circuits

I'm now going to plot some QK circuits. I expect to see the following:

* Head (0.1 off token `a`) and (0.2 on tokens `[a, b, g]`) will have a positive stripe, for the (query-embedding) x (key-embedding) QK circuit.
* Head 0.0, and (0.2 everywhere except `[a, b, g]`) will attend more to early tokens, i.e. they'll have a smooth gradient over source positions for both the (query-embedding) x (key-pos-embed) and (query-pos-embed) x (key-pos-embed) QK circuits.

To be safe, I also wanted to make a bar chart of the mean & std of the layernorm scale factors which I'm using in this computation, to make sure they aren't implementing any complicated logic (they seem not to be).

```python
W_pos_labels = [str(i) for i in range(model.cfg.n_ctx)]

# Check layernorm scale factor mean & std dev, verify that std dev is small
scale = cache["scale", 0, "ln1"][:, :, 0, 0] # shape (batch, seq)
df = pd.DataFrame({
    "std": scale.std(0).cpu().numpy(),
    "mean": scale.mean(0).cpu().numpy(),
})
px.bar(
    df, 
    title="Mean & std of layernorm before first attn layer", 
    template="simple_white", width=600, height=400, barmode="group"
).show()

W_QK: Tensor = model.W_Q[0] @ model.W_K[0].transpose(-1, -2) / (model.cfg.d_head ** 0.5)

W_E_scaled = model.W_E / scale.mean()
W_pos_scaled = model.W_pos / scale.mean(dim=0).unsqueeze(-1)

W_Qemb_Kemb = W_E_scaled @ W_QK @ W_E_scaled.T
W_Qboth_Kpos = t.concat([W_E_scaled, W_pos_scaled]) @ W_QK @ W_pos_scaled.T
# Apply causal masking
W_Qboth_Kpos[:, -len(W_pos_labels):].masked_fill_(t.triu(t.ones_like(W_Qboth_Kpos[:, -len(W_pos_labels):]), diagonal=1).bool(), float("-inf"))

imshow(
    W_Qemb_Kemb,
    facet_col = 0,
    facet_labels = [f"0.{head}" for head in range(model.cfg.n_heads)],
    title = f"Query = WE, Key = WE",
    labels = {"x": "Source", "y": "Dest"},
    x = dataset.vocab,
    y = dataset.vocab,
    height = 400,
    width = 750,
)
imshow(
    W_Qboth_Kpos,
    facet_col = 0,
    facet_labels = [f"0.{head}" for head in range(model.cfg.n_heads)],
    title = f"Query = WE & Wpos, Key = Wpos",
    labels = {"x": "Source", "y": "Dest"},
    x = W_pos_labels,
    y = dataset.vocab + W_pos_labels,
    height = 620,
    width = 1070,
)
```
""", unsafe_allow_html=True)
    
    with open(unique_char_dir / "fig_ln_std.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=400)
    with open(unique_char_dir / "fig_Qe_Ke.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=400)
    with open(unique_char_dir / "fig_Qep_Kp.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=640)

    st.markdown(
r"""
## Conclusions

This pretty fits with both the two expectations I had in the previous section. The query-side positional embeddings actually seem to have a slight bias towards attending to later positions, but it looks like this is dominated by the effect from the query-side token embeddings (which show a stronger "attend to earlier positions" effect). Also, note that 0.1 has a bias against self-attention, which makes sense given its role as a DTH.

One other observation - heads 0.0 and 0.2 self-attending strongly at position 1 stands out here. This is a good indication that V-composition between these two heads & some heads in layer 1 is boosting tokens, because "boost logits for the very first non-null token in the sequence, mostly at the first position but also at the positions that come after" is a very easy and helpful heuristic to learn. In fact, we might speculate this was one of the first things the model learned (after the obvious "predict null character at the start of the sequence"). The algorithm proposed at the start (all heads in layer 0 acting as duplicate token heads, and heads in layer 1 attending to the first non-duplicated token) might actually have achieved better global loss properties. But if the model learned this heuristic early on, and a consequence of this heuristic is positive virtual copying circuits forming between (0.0, 0.2) and heads in layer 1, then it might have no choice but to implement the version of the algorithm we see here.

**Exercise to the reader** - can you find evidence for/against this claim? You can find all the details of training, including random seeds, in the notebook `august23_unique_char/training_model.ipynb`. Is this heuristic one of the first things the model learns? If you force the model not to learn this heuristic during training (e.g. by adding a permanent hook to make sure heads in layer 0 never self-attend), does the model learn a different algorithm more like the one proposed at the start?

# Direct Logit Attribution

The last thing I'll do here (before moving onto some adversarial examples) is write a function to decompose the model's logit attribution by path through the model. Specifically, I can split it up into the same 16 paths as we saw in the "full virtual OV circuits" heatmap earlier. This will help to see whether the theories I've proposed about the model's behaviour are correct.

It wasn't obvious what kind of visualisation was appropriate here. If I focused on a single sequence, then there are 3 dimensions I care about: the destination position, the path through the model, and which token is being suppressed / boosted. I experimented with heatmaps using each of these three dimensions as a facet column, and ended up settling on using the latter of these as the facet column - that way it's easier to compare different paths, at different points in the sequence (and because there's usually only a handful of tokens I care about the logit attribution of, at any given point).

One last note - I subtracted the mean DLA for every path. This turned out to be important, because there are a few effects we need to control for (in particular, the direct effect of head `1.0` in `c` even when it's not in the sequence). This is why the first column is all zeros (this doesn't mean the head is unable to predict `?` as the first character!). I've also allowed this mean to be returned as a tensor and used as input rather than a boolean, in case I want to subtract the mean for a much smaller dataset.

```python
def dla_imshow(
    dataset: UniqueCharDataset,
    cache: ActivationCache,
    batch_idx: int,
    str_tok: Union[str, List[str]],
    subtract_mean: Union[bool, Tensor] = True,
):
    # ! Get DLA from the direct paths & paths through just heads in layer 0
    resid_decomposed = t.stack([
        cache["embed"] + cache["pos_embed"],
        *[cache["result", 0][:, :, head] for head in range(3)]
    ], dim=1)
    assert resid_decomposed.shape == (len(dataset), 4, 20, model.cfg.d_model), resid_decomposed.shape
    t.testing.assert_close(resid_decomposed.sum(1) + model.b_O[0], cache["resid_post", 0])

    dla = (resid_decomposed / cache["scale"].unsqueeze(1)) @ model.W_U
    assert dla.shape == (len(dataset), 4, 20, model.cfg.d_vocab), dla.shape

    # ! Get DLA from paths through layer 1
    resid_decomposed_post_W_OV = einops.einsum(
        (resid_decomposed / cache["scale", 0, "ln1"][:, None, :, 0]),
        model.W_V[1] @ model.W_O[1],
        "batch decomp seqK d_model, head d_model d_model_out -> batch decomp seqK head d_model_out"
    )
    resid_decomposed_post_attn = einops.einsum(
        resid_decomposed_post_W_OV,
        cache["pattern", 1],
        "batch decomp seqK head d_model, batch head seqQ seqK -> batch decomp seqQ head d_model"
    )
    new_dla = (resid_decomposed_post_attn / cache["scale"][:, None, :, None]) @ model.W_U
    dla = t.concat([
        dla,
        einops.rearrange(new_dla, "batch decomp seq head vocab -> batch (decomp head) seq vocab")
    ], dim=1)

    # ! Get DLA for batch_idx, subtract mean baseline, optionally return the mean
    dla_mean = dla.mean(0)
    if isinstance(subtract_mean, Tensor):
        dla = dla[batch_idx] - subtract_mean
    elif subtract_mean:
        dla = dla[batch_idx] - dla_mean
    else:
        dla = dla[batch_idx]

    # ! Plot everything
    if isinstance(str_tok, str):
        str_tok = [str_tok]
        kwargs = dict(
            title = f"Direct Logit Attribution by path, for token {str_tok[0]!r}",
            height = 550,
            width = 700,
        )
    else:
        assert len(str_tok) % 2 == 0, "Odd numbers mess up figure order for some reason"
        kwargs = dict(
            title = "Direct Logit Attribution by path",
            facet_col = -1,
            facet_labels = [f"DLA for token {s!r}" for s in str_tok],
            height = 100 + 450 * int(len(str_tok) / 2),
            width = 1250,
            facet_col_wrap = 2,
        )
    toks = [dataset.vocab.index(tok) for tok in str_tok]
    layer0 = [" "] + [f"0.{i} " for i in range(3)]
    layer1 = [f"1.{i} " for i in range(3)]
    imshow(
        dla[:, :, toks].squeeze(),
        x = [f"{s}({i})" for i, s in enumerate(dataset.str_toks[batch_idx])],
        y = layer0 + [f"{c0}➔ {c1}".lstrip(" ➔ ") for c0 in layer0 for c1 in layer1],
        # margin = dict.fromkeys("tblr", 40),
        aspect = "equal",
        text_auto = ".0f",
        **kwargs,
    )
    if isinstance(subtract_mean, bool) and subtract_mean:
        return dla_mean


print(f"Seq = {''.join(dataset.str_toks[0])}, Target = {''.join(dataset.str_tok_labels[0])}")

dla_mean = dla_imshow(
    dataset,
    cache,
    batch_idx = 0, 
    str_tok = ["c", "g"],
    subtract_mean = True,
)
```

<div style='font-family:monospace; font-size:15px;'>
Seq = ?chgegfaeadieaebcffh, Target = ?ccccccccccccccchhhd
</div><br>
""", unsafe_allow_html=True)
    
    with open(unique_char_dir / "fig_dla.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=620, width=1000, scrolling=True)

    st.markdown(
r"""
Playing around with these plots for a while, I concluded that they pretty much fit my expectations. The paths which are doing boosting and suppression are almost always the ones I'd expect from the OV composition plot.

For example, take the plot above, which shows the attribution for `[c, g]` in the very first sequence. Consider the attribution for `c`:

* At position 1, the paths (0.0, 0.2) ➔ 1.0 boost `c`, and (0.1, direct) ➔ 1.0 suppress it (the former is stronger, presumably because the first character in 0.0 and 0.2 strongly self-attends). This fits with our virtual OV circuits plot. 
* After position 1, the main positive attribution paths are (0.0, 0.2) ➔ 1.0, as expected.
* Once `c` becomes duplicated for the first time, the negative attribution from (0.1, direct) ➔ 1.0 outweighs the positive attribution. This makes sense, because once `c` is duplicated head 0.1 and 1.0 will both attend more to the duplicated `c` (neither of which boosts the positive paths for `c`, since the duplicated `c` doesn't attend back to the first instance of `c` in 0.0 or 0.2).
        
Now consider the attribution for `g`:

* The path (0.0 ➔ 1.2) boosts `g` at positions 3 & 4, as expected from the virtual OV circuits plot.
* The paths (direct, 0.2 ➔ 1.2) kick in after the duplicated token to suppress `g` for the rest of the sequence (as well as 0.1 ➔ 1.2, but this one is weaker) - again, this is the path we expect to see.

# Final summary

Some layer 0 heads (0.1 everywhere except `a`, and 0.2 on `[a, b, g]`) are duplicate token heads; they're composing with layer 1 heads to cause those heads to attend to & suppress duplicate tokens. This is done both with K-composition (heads in layer 1 attend more to duplicated tokens), and V-composition (the actual outputs of the DTHs are used as value input to heads in layer 1 to suppress duplicated tokens).

All other layer 0 head paths are involved in boosting, rather than suppression. They attend to early tokens, which are not the same as the current destination token. Their outputs are used as value input to heads in layer 1 to boost these tokens. 

Layer 1 heads split their functionality across the vocabulary. 1.0 handles boosting / suppression for `[a, c]`, 1.1 handles `[d, e, f, j]`, and 1.2 handles `[b, g, h, i]`. These sets are disjoint, and their union is the whole vocabulary. This makes sense, because layer 1 attention is a finite resource, and it has to be used to suppress every duplicated token in the sequence (missing even one duplicated token could cause the model to make an incorrect classification).

# Adversarial examples

To make things interesting, I'll try and find an example where the model thinks the answer is X rather than Y (as opposed to thinking that there is a unique solution when there isn't, or vice-versa).

Firstly, it seems like a good idea to overload single heads if we can (think of overloading chess pieces!). Head 1.0 manages `ac` (2 tokens), 1.1 manages `defj` (4 tokens), and 1.2 manages `bghi` (4 tokens). The latter heads have more responsibilities, so we should try overloading one of them.

Secondly, we don't want the correct token to be the one at the first non-null position - that would be too easy! Heads like 0.0 and 0.2 strongly self-attend at the first non-null position, and then heads in layer 1 attend to this position in order to boost those logits. We need to put a few duplicated tokens first in the sequence.

Thirdly, we should have 2 unique tokens right next to each other, in the hope that the model will think the second one is the correct answer rather than the first one. We saw a smooth gradient with the full QK circuits (when the key-side circuit was positional), so the differences between adjacent tokens should be minimal.

After searching for a bit, I found the example below. We're overloading head 1.2, by including duplicated tokens `[g, b, i]` before a non-duplicated `h` and non-duplicated `a`. Head 1.0 is able to boost `a` because it's not overloaded, but head 1.2 is unable to boost `h` because it's already attending to & suppressing the duplicated tokens `[g, b, i]`. There are a few more examples like this you can create if you play around with the exact order and identity of tokens.

```python
class CustomDataset(UniqueCharDataset):
        
    def __init__(
        self,
        tokens: Union[Int[Tensor, "batch seq"], Callable],
        size: Optional[int] = None,
        vocab: List[str] = list("abcdefghij"), 
        seq_len: int = 20,
        seed: int = 42
    ):
        
        self.vocab = vocab + ["?"]
        self.null_tok = len(vocab)
        if isinstance(tokens, Tensor):
            self.size = tokens.shape[0]
        else:
            assert size is not None
            self.size = size
        t.manual_seed(seed)

        # Generate our sequences
        if isinstance(tokens, t.Tensor):
            self.toks = tokens
        else:
            self.toks = tokens(self.size, seq_len, self.null_tok)
        self.str_toks = [
            [self.vocab[tok] for tok in seq]
            for seq in self.toks
        ]

        # Generate our labels (i.e. the identity of the first non-repeating character in each sequence)
        self.labels = find_first_unique(self.toks[:, 1:], self.null_tok)
        self.labels = t.cat([
            t.full((self.size, 1), fill_value=self.null_tok),
            self.labels
        ], dim=1)
        self.str_tok_labels = [
            [self.vocab[tok] for tok in seq]
            for seq in self.labels
        ]


str_toks = "?ggbbiihaggbigbigbig"
toks = t.tensor([[dataset.vocab.index(tok) for tok in str_toks]])

advex_dataset = CustomDataset(tokens=toks)

advex_logits, advex_cache = model.run_with_cache(advex_dataset.toks)
advex_logprobs = advex_logits.squeeze().log_softmax(-1).T
advex_probs = advex_logits.squeeze().softmax(-1).T

print(f"Seq = {''.join(advex_dataset.str_toks[0])}, Target = {''.join(advex_dataset.str_tok_labels[0])}")

imshow(
    advex_probs,
    y=advex_dataset.vocab,
    x=[f"{s}({j})" for j, s in enumerate(str_toks)],
    labels={"x": "Position", "y": "Predicted token"},
    title="Probabilities for adversarial example",
    width=800,
    text=[
        ["〇" if str_tok == correct_str_tok else "" for correct_str_tok in advex_dataset.str_tok_labels[0]]
        for str_tok in advex_dataset.vocab
    ],
)
```""", unsafe_allow_html=True)
    
    with open(unique_char_dir / "fig_advex.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=400)

    st.markdown(
r"""
Verify that head 1.2 is attending strongly to the duplicated `gbi` tokens, less to to `h` and those after it:

```python
cv.attention.from_cache(
    advex_cache,
    tokens = list(str_toks),
    attention_type = "standard",
)
```
""", unsafe_allow_html=True)
    
    with open(unique_char_dir / "fig_attn_2.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=400)

    st.markdown(
r"""
# Remaining questions / notes / things not discussed

## Null character

I've not discussed how the model predicts the null character yet, because I didn't consider it a crucial part of the model's functionality. Some more investigation leads me to the following hypothesis:

* The model predicts `?` at the first position because every attention head has to self-attend here. Looking at DLA without subtracting the mean shows that almost every path contributes a small positive amount to `?`.
* In later positions, `?` is predicted in much the same way other tokens are predicted: all duplicated tokens are strongly suppressed, and `?` is boosted mainly via V-composition paths. This effect is weaker than for other tokens in the vocabulary - which it has to be, because if any non-duplicated token exists then it needs to dominate the boosting of `?`.
* Unlike the other characters in the vocabulary, `?` doesn't have a dedicated head in layer 1. 

## Layer 1 QK circuits

There are some interesting patterns here, somewhat analogous to the patterns in the layer 0 QK circuits. I've not discussed them here though, because I don't consider them critical to understand how this model functions.

## How can the model predict a token with high probability, without ever attending to it with high probability?

I'm including a short answer to this question, because it's something which confused me a lot when I started looking at this model. 

Consider a sequence like `?aab...c` as an example. How can the model correctly predict `b` at position `c`? The answer, in short - **in heads 0.0 and 0.2, all the tokens between `b` and `c` will slightly attend to `b`. Then in head 1.2, `c` will attend to these intermediate tokens, and these virtual OV circuits will boost `b`.** Also, the duplicate token head 0.1 makes sure `a` is very suppressed, so that `b` will be predicted with highest probability.""")


def section_0_september():

    st.sidebar.markdown(
r"""

## Table of Contents

<ul class="contents">
    <li><a class='contents-el' href='#prerequisites'>Prerequisites</a></li>
    <li><a class='contents-el' href='#difficulty'>Difficulty</a></li>
    <li><a class='contents-el' href='#motivation'>Motivation</a></li>
    <li><a class='contents-el' href='#logistics'>Logistics</a></li>
    <li><a class='contents-el' href='#what-counts-as-a-solution'>What counts as a solution?</a></li>
    <li><a class='contents-el' href='#setup'>Setup</a></li>
    <li><a class='contents-el' href='#task-dataset'>Task & Dataset</a></li>
    <li><a class='contents-el' href='#model'>Model</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(
r"""
# Monthly Algorithmic Challenge (September 2023): Sum Of Two Numbers

### Colab: [problem](https://colab.research.google.com/drive/1770X6JLjizn5GLFPoLw3wWx44TXxQVg5) | [solutions](https://colab.research.google.com/drive/1HBec9II1Ozt_1i6lE9uWbGa6v1VKUC6y)

This post is the third in the sequence of monthly mechanistic interpretability challenges. They are designed in the spirit of [Stephen Casper's challenges](https://www.lesswrong.com/posts/KSHqLzQscwJnv44T8/eis-vii-a-challenge-for-mechanists), but with the more specific aim of working well in the context of the rest of the ARENA material, and helping people put into practice all the things they've learned so far.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/machines.png" width="350">

## Prerequisites

The following ARENA material should be considered essential:

* **[1.1] Transformer from scratch** (sections 1-3)
* **[1.2] Intro to Mech Interp** (sections 1-3)

The following material isn't essential, but is recommended:

* **[1.2] Intro to Mech Interp** (section 4)
* **July's Algorithmic Challenge - writeup** (on the sidebar of this page)

## Difficulty

This problem is slightly easier than the September problem. I expect solutions to rely less on high-level ideas like path decomposition, relative to last month's problem. It is still a more difficult problem than the July problem.

## Motivation

Neel Nanda's post [200 COP in MI: Interpreting Algorithmic Problems](https://www.lesswrong.com/posts/ejtFsvyhRkMofKAFy/200-cop-in-mi-interpreting-algorithmic-problems) does a good job explaining the motivation behind solving algorithmic problems such as these. I'd strongly recommend reading the whole post, because it also gives some high-level advice for approaching such problems.

The main purpose of these challenges isn't to break new ground in mech interp, rather they're designed to help you practice using & develop better understanding for standard MI tools (e.g. interpreting attention, direct logit attribution), and more generally working with libraries like TransformerLens.

Also, they're hopefully pretty fun, because why shouldn't we have some fun while we're learning?

## Logistics

The solution to this problem will be published on this page in the first few days of October, at the same time as the next problem in the sequence. There will also be an associated LessWrong post.

If you try to interpret this model, you can send your attempt in any of the following formats:

* Colab notebook,
* GitHub repo (e.g. with ipynb or markdown file explaining results),
* Google Doc (with screenshots and explanations),
* or any other sensible format.

You can send your attempt to me (Callum McDougall) via any of the following methods:

* The [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), via a direct message to me
* My personal email: `cal.s.mcdougall@gmail.com`
* LessWrong message ([here](https://www.lesswrong.com/users/themcdouglas) is my user)

**I'll feature the names of everyone who sends me a solution on this website, and also give a shout out to the best solutions.**

Please don't discuss specific things you've found about this model until the challenge is over (although you can discuss general strategies and techniques, and you're also welcome to work in a group if you'd like). The deadline for this problem will be the end of this month, i.e. 31st August.

## What counts as a solution?

Going through the solutions for the previous problems in the sequence (July: Palindromes & August: First Unique Character) as well as the exercises in **[1.4] Balanced Bracket Classifier** should give you a good idea of what I'm looking for. In particular, I'd expect you to:

* Describe a mechanism for how the model solves the task, in the form of the QK and OV circuits of various attention heads (and possibly any other mechanisms the model uses, e.g. the direct path, or nonlinear effects from layernorm),
* Provide evidence for your mechanism, e.g. with tools like attention plots, targeted ablation / patching, or direct logit attribution.
* (Optional) Include additional detail, e.g. identifying the subspaces that the model uses for certain forms of information transmission, or using your understanding of the model's behaviour to construct adversarial examples.

# Setup

```python
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from pathlib import Path

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "monthly_algorithmic_problems" / "september23_sum"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from monthly_algorithmic_problems.september23_sum.dataset import SumDataset
from monthly_algorithmic_problems.september23_sum.model import create_model
from plotly_utils import hist, bar, imshow

device = t.device("cuda" if t.cuda.is_available() else "cpu")
```

## Task & Dataset

The problem for this month is interpreting a model which has been trained to perform simple addition. The model was fed input in the form of a sequence of digits (plus special + and = characters with token ids 10 and 11), and was tasked with predicting the sum of digits one sequence position before they would appear. Cross entropy loss was only applied to these four token positions, so the model's output at other sequence positions is meaningless.
                
<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/predictions.png" width="600">

All the left-hand numbers are below 5000, so we don't have to worry about carrying past the thousands digit.

Here is an example of what this dataset looks like:

```python
dataset = SumDataset(size=1, num_digits=4, seed=42)

print(dataset[0].tolist()) # tokens, for passing into model
print("".join(dataset.str_toks[0])) # string tokens, for printing
```

<div style='font-family:monospace; font-size:15px;'>
[2, 7, 6, 4, 10, 1, 5, 0, 4, 11, 4, 2, 6, 8]<br>
2764+1504=4268
</div><br>

The relevant files can be found at:

```
chapter1_transformers/
└── exercises/
    └── monthly_algorithmic_problems/
        └── september23_sum/
            ├── model.py               # code to create the model
            ├── dataset.py             # code to define the dataset
            ├── training.py            # code to training the model
            └── training_model.ipynb   # actual training script
```

We've given you the class `SumDataset` to store your data, as you can see above. You can slice this object to get tokens, or use the `str_toks` attribute (a list of lists of strings).

## Model

Our model was trained by minimising cross-entropy loss between its predictions and the true labels, at the four positions of the sum's digits. You can inspect the notebook `training_model.ipynb` to see how it was trained. I used the version of the model which achieved highest accuracy over 100 epochs (accuracy ~100%).

The model is is a 2-layer transformer with 3 attention heads, and causal attention. It includes layernorm, but no MLP layers. You can load it in as follows:

```python
filename = section_dir / "sum_model.pt"

model = create_model(
    num_digits=4,
    seed=0,
    d_model=48,
    d_head=24,
    n_layers=2,
    n_heads=3,
    normalization_type="LN",
    d_mlp=None
)

state_dict = t.load(filename)

state_dict = model.center_writing_weights(t.load(filename))
state_dict = model.center_unembed(state_dict)
state_dict = model.fold_layer_norm(state_dict)
state_dict = model.fold_value_biases(state_dict)
model.load_state_dict(state_dict, strict=False);
```

The code to process the state dictionary is a bit messy, but it's necessary to make sure the model is easy to work with. For instance, if you inspect the model's parameters, you'll see that `model.ln_final.w` is a vector of 1s, and `model.ln_final.b` is a vector of 0s (because the weight and bias have been folded into the unembedding).

```python
print("ln_final weight: ", model.ln_final.w)
print("\nln_final, bias: ", model.ln_final.b)
```

<details>
<summary>Aside - the other weight processing parameters</summary>

Here's some more code to verify that our weights processing worked, in other words:

* The unembedding matrix has mean zero over both its input dimension (`d_model`) and output dimension (`d_vocab`)
* All writing weights (i.e. `b_O`, `W_O`, and both embeddings) have mean zero over their output dimension (`d_model`)
* The value biases `b_V` are zero (because these can just be folded into the output biases `b_O`)

```python
W_U_mean_over_input = einops.reduce(model.W_U, "d_model d_vocab -> d_model", "mean")
t.testing.assert_close(W_U_mean_over_input, t.zeros_like(W_U_mean_over_input))

W_U_mean_over_output = einops.reduce(model.W_U, "d_model d_vocab -> d_vocab", "mean")
t.testing.assert_close(W_U_mean_over_output, t.zeros_like(W_U_mean_over_output))

W_O_mean_over_output = einops.reduce(model.W_O, "layer head d_head d_model -> layer head d_head", "mean")
t.testing.assert_close(W_O_mean_over_output, t.zeros_like(W_O_mean_over_output))

b_O_mean_over_output = einops.reduce(model.b_O, "layer d_model -> layer", "mean")
t.testing.assert_close(b_O_mean_over_output, t.zeros_like(b_O_mean_over_output))

W_E_mean_over_output = einops.reduce(model.W_E, "token d_model -> token", "mean")
t.testing.assert_close(W_E_mean_over_output, t.zeros_like(W_E_mean_over_output))

W_pos_mean_over_output = einops.reduce(model.W_pos, "position d_model -> position", "mean")
t.testing.assert_close(W_pos_mean_over_output, t.zeros_like(W_pos_mean_over_output))

b_V = model.b_V
t.testing.assert_close(b_V, t.zeros_like(b_V))
```

</details>

A demonstration of the model working:

```python
dataset = SumDataset(size=1000, num_digits=4, seed=42).to(device)

targets = dataset.toks[:, -4:]

logits, cache = model.run_with_cache(dataset.toks)
logits = logits[:, -5:-1]

logprobs = logits.log_softmax(-1) # [batch seq_len vocab_out]
probs = logprobs.softmax(-1)

# Library developed for easier indexing - feel free not to use
%pip install git+https://github.com/callummcdougall/eindex.git
from eindex import eindex

logprobs_correct = eindex(logprobs, targets, "batch seq [batch seq]")
probs_correct = eindex(probs, targets, "batch seq [batch seq]")

print(f"Average cross entropy loss: {-logprobs_correct.mean().item():.3f}")
print(f"Mean probability on correct label: {probs_correct.mean():.3f}")
print(f"Median probability on correct label: {probs_correct.median():.3f}")
print(f"Min probability on correct label: {probs_correct.min():.3f}")
```

<div style='font-family:monospace; font-size:15px;'>
Average cross entropy loss: 0.017<br>
Average probability on correct label: 0.988<br>
Min probability on correct label: 0.001
</div><br>

And a visualisation of its probability output for a single sequence:

```python
def show(i):

    imshow(
        probs[i].T,
        y=dataset.vocab,
        x=[f"{dataset.str_toks[i][j]}<br><sub>({j})</sub>" for j in range(9, 13)],
        labels={"x": "Token", "y": "Vocab"},
        xaxis_tickangle=0,
        title=f"Sample model probabilities:<br>{''.join(dataset.str_toks[i])}",
        text=[
            ["〇" if (str_tok == target) else "" for target in dataset.str_toks[i][-4:]]
            for str_tok in dataset.vocab
        ],
        width=400,
        height=550,
    )

show(0)
```
""", unsafe_allow_html=True)
    
    with open(sum_dir / "fig_demo.html", 'r', encoding='utf-8') as f: fig1 = f.read()
    st_html(fig1, height=500)

    st.markdown(r"""
If you want some guidance on how to get started, I'd recommend reading the solutions for the July problem - I expect there to be a lot of overlap in the best way to tackle these two problems. You can also reuse some of that code!

Note - although this model was trained for long enough to get loss close to zero (you can test this for yourself), it's not perfect. There are some weaknesses that the model has which might make it vulnerable to adversarial examples, and I've decided to leave these in. The model is still very good at its intended task, and the main focus of this challenge is on figuring out how it solves the task, not dissecting the situations where it fails. However, you might find that the adversarial examples help you understand the model better.

Best of luck! 🎈

""", unsafe_allow_html=True)


def section_1_september():
    st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li><a class='contents-el' href='#summary-of-how-the-model-works'>Summary of how the model works</a></li>
    <li><a class='contents-el' href='#notation'>Notation</a></li>
    <li><a class='contents-el' href='#some-initial-notes'>Some initial notes</a></li>
    <li><a class='contents-el' href='#first-pass-attention-patterns-and-ablations'>First pass - attention patterns, and ablations</a></li>
    <li><a class='contents-el' href='#unembedding-matrix-structure'>Unembedding matrix structure</a></li>
    <li><a class='contents-el' href='#ablation-experiments-to-test-the-carry-information-theory'>Ablation experiments to test the "carry information" theory</a></li>
    <li><a class='contents-el' href='#linear-probes'>Linear probes</a></li>
    <li><a class='contents-el' href='#final-summary'>Final summary</a></li>
</ul></li>""", unsafe_allow_html=True)
    
    st.markdown(
r"""
# Monthly Algorithmic Challenge (September 2023): Solutions

We assume you've run all the setup code from the previous page "[September] Sum of Two Numbers". Here's all the new setup code you'll need:
                
```python
%pip install git+https://github.com/callummcdougall/eindex.git

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
from typing import List, Optional, Literal, cast
import torch as t
from torch import Tensor
import plotly.express as px
import einops
from jaxtyping import Float
from pathlib import Path
import pandas as pd
import circuitsvis as cv
from transformer_lens import ActivationCache
from eindex import eindex
import plotly.express as px
import torch
from transformer_lens import utils
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint
import plotly.graph_objects as go
from plotly.subplots import make_subplots

t.set_grad_enabled(False)

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "monthly_algorithmic_problems" / "august23_unique_char"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from monthly_algorithmic_problems.september23_sum.model import create_model
from monthly_algorithmic_problems.september23_sum.dataset import SumDataset
from plotly_utils import imshow

device = t.device("cuda" if t.cuda.is_available() else "cpu")

dataset = SumDataset(size=1000, num_digits=4, seed=42).to(device)
N = len(dataset)

# Print some output
for toks, str_toks in zip(dataset, dataset.str_toks[:5]):
    print("".join(str_toks))

# Define some useful objects
LABELS_STR = ['A0', 'A1', 'A2', 'A3', '+', 'B0', 'B1', 'B2', 'B3', '=', 'C0', 'C1', 'C2', 'C3']
LABELS_HTML = [f"A<sub>{i}</sub>" for i in range(4)] + ["+"] + [f"B<sub>{i}</sub>" for i in range(4)] + ["="] + [f"C<sub>{i}</sub>" for i in range(4)]
LABELS_DICT = {label: i for i, label in enumerate(LABELS_STR)}
```

And for getting some activations:

```python
targets = dataset.toks[:, -4:]

logits, cache = model.run_with_cache(dataset.toks)
logits: Tensor = logits[:, -5:-1]

logprobs = logits.log_softmax(-1) # [batch seq_len vocab_out]
probs = logprobs.softmax(-1)

batch_size, seq_len = dataset.toks.shape
logprobs_correct = eindex(logprobs, targets, "batch seq [batch seq]")
probs_correct = eindex(probs, targets, "batch seq [batch seq]")

avg_cross_entropy_loss = -logprobs_correct.mean().item()

print(f"Average cross entropy loss: {avg_cross_entropy_loss:.3f}")
print(f"Mean probability on correct label: {probs_correct.mean():.3f}")
print(f"Median probability on correct label: {probs_correct.median():.3f}")
print(f"Min probability on correct label: {probs_correct.min():.3f}")
```

Output:

<div style='font-family:monospace; font-size:15px;'>
Average cross entropy loss: 0.007<br>
Mean probability on correct label: 0.993<br>
Median probability on correct label: 0.996<br>
Min probability on correct label: 0.759
</div><br>

## Summary of how the model works

To calculate each digit `Ci`, we require 2 components - the **sum** and the **carry**. The formula for `Ci` is `(sum + int(carry == True)) % 10`, where `sum` is the sum of digits `Ai + Bi`, and `carry` is whether `A(i+1) + B(i+1) >= 10`. (This ignores issues of carrying digits multiple times, which I won't discuss in this solution.)

We calculate the carry by using the hierarchy $0 > 9 > 1 > 8 > ... > 4 > 5$. An attention head in layer 0 will attend to the first number in this hierarchy that it sees, and if that number is $\geq 5$ then that means the digit will be carried. There are also some layer 0 attention heads which store the sum information in certain sequence positions - either by attending uniformly to both digits, or by following the reverse hierarchy so it can additively combine with something that follows the hierarchy. Below is a visualisation of the QK circuits for the layer 0 attention heads at the positions which are storing this "carry" information, to show how they're implementing the hierarchy:

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/hierarchy.png" width="650">

At the end of layer 0, the sum information is stored in the residual stream as points around a circle traced out by two vectors, parameterized by an angle $\theta$. The carry information is stored in the residual stream as a single direction.

The model manages to store the sum of the two digits modulo 10 in a circular way by the end of layer 0 (although it's not stored in exactly the same way it will be at the end of the model). We might guess the model takes advantage of some trig identities to do this, although I didn't have time to verify this conclusively.

The heads in layer 1 mostly process this information by self-attending. They don't seem as important as heads `0.1` and `0.2` (measured in terms of loss after ablation), and it seems likely they're mainly clearing up some of the representations learned by the layer 0 heads (and dealing with logic about when to carry digits multiple times).

By the end of layer 1, the residual stream is parameterized by a single value: the angle $\theta$. The digits from 0-9 are evenly spaced around the unit circle, and the model's prediction depends on which angle they're closest to. Two visualisations of this are shown below: (1) the singular value decomposition of the unembedding matrix, and (2) the residual stream projected onto these first two singular directions.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/sing.png" width="700">
<br>
<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/two-sing.png" width="600">

## Notation

We'll refer to the sequence positions as `A0`, `A1`, `A2`, `A3`, `+`, `B0`, `B1`, `B2`, `B3`, `=`, `C0`, `C1`, `C2`, `C3`.

Usually, this will refer to those sequence positions, but sometimes it'll refer to the tokens at those sequence positions (hopefully it'll be clear which one we mean from context).

## Some initial notes

* There are 2 different parts of figuring out each digit: adding two things together, and figuring out whether the digit needs to be incremented
* I expect the problem to be easiest when looking at the units digit, because there's no incrementation to worry about
* I expect all other digits to be implementing something of the form "do the units digit alg, plus do other things to compute whether a digit needs to be carried"
* To make life simpler, I could create a dataset which only contains digits that don't require any carrying (e.g. all digits were uniform between 0 and 4 inclusive)

Things I expect to see:

* Attention patterns
    1. There will be a head / heads in layer 0, which attend from X back to the two digits that are being added together to create X
    2. There will be a head / heads in layer 1, which have a more difficult job: figuring out incrementation
* Full matrices: QK
    1. The layer 0 head mentioned above will have a QK circuit that is only a function of position (it'll deliberately ignore token information cause it needs to always get an even split)
* Other things
    * Neel's modular arithmetic model used Fourier stuff to implement modular arithmetic. I'm essentially doing modular arithmetic here too, since I'm calculating the sum of 2 digits modulo 10. It's possible this is just done via some memorization system (cause it's a much smaller dataset than Neel's model was trained with), but I'd weakly guess Fourier stuff is involved.

## First pass - attention patterns, and ablations

First experiments - we'll look at the attention patterns, then narrow in on the heads which are doing the thing we think must happen (i.e. equal attention back to both digits). Do we see what we expect to see?

Before doing attention patterns though, I'll plot the mean attention paid from to/from each combination of tokens. I'm expecting to see some patterns where the avg attention is approximately 0.5 for each of a pair of digits from the numbers being added together (because addition is a symmetric operation). We might guess that sequence positions in heads in layer 0 which don't have this "uniform average" aren't actually doing important things.

```python
attn = t.concat([cache["pattern", layer] for layer in range(model.cfg.n_layers)], dim=1) # [batch heads seqQ seqK]

imshow(
    attn.mean(0),
    facet_col=0,
    facet_labels=[f"{layer}.{head}" for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)],
    facet_col_wrap=model.cfg.n_heads,
    height=700,
    width=900,
    x=LABELS_STR,
    y=LABELS_STR,
)
```

""", unsafe_allow_html=True)

    with open(sum_dir / "fig_attn.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=700)

    st.markdown(
r"""
We can see a few positions in layer 0 which are paying close to 0.5 average attention to each of some two digits being added together (e.g. positions `=` and `C0` in head `0.2`). We don't see any patterns like this in layer 1.

Now, let's inspect attention patterns in actual examples.
""", unsafe_allow_html=True)

    with open(sum_dir / "fig_cv.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=625)

    st.markdown(
r"""
Before we start over-interpreting patterns, let's run some mean ablations of different heads to see which ones matter. I've added an argument `mode` which can be set to either "read" or "write" (i.e. we can ablate either the head's output or its input).

```python
def get_loss_from_ablating_head(layer: int, head: int, seq_pos: int, mode: Literal["write", "read"] = "write"):
    '''
    Calculates the loss from mean-ablating a head at a particular sequence position, over
    each of the 4 prediction sequence positions.

    By default `mode='write'`, i.e. we're ablating the head's output. We can also ablate
    the head's value input with `mode='read'`.
    '''

    def hook_fn(activation: Float[Tensor, "batch seq nheads d"], hook: HookPoint):
        activation_mean: Float[Tensor, "d_model"] = cache[hook.name][:, seq_pos, head].mean(0)
        activation[:, seq_pos, head] = activation_mean
        return activation
        
    if mode == "write":
        hook_names = [utils.get_act_name("result", layer)]
    elif mode == "read":
        hook_names = [utils.get_act_name(name, layer) for name in "qkv"]
    
    model.reset_hooks()
    logits_orig = model(dataset.toks)
    logprobs_orig = logits_orig[:, -5:-1].log_softmax(-1)
    logits_ablated = model.run_with_hooks(dataset.toks, fwd_hooks=[(lambda name: name in hook_names, hook_fn)])
    logprobs_ablated = logits_ablated[:, -5:-1].log_softmax(-1)

    targets = dataset.toks[:, -4:]

    # For each output position we're trying to predict, we measure the difference in loss
    loss_diffs = []
    for i in range(4):
        loss_orig = -logprobs_orig[range(N),  i, targets[:,  i]]
        loss_ablated = -logprobs_ablated[range(N),  i, targets[:,  i]]
        loss_diff = (loss_ablated - loss_orig).mean().item()
        loss_diffs.append(loss_diff)

    return t.tensor(loss_diffs)


def plot_all_ablation_loss(layer: int, mode: Literal["write", "read"] = "write"):

    loss_diffs = t.zeros(model.cfg.n_heads, model.cfg.n_ctx, 4)

    for head in range(model.cfg.n_heads):
        for seq_pos in range(model.cfg.n_ctx):
            loss_diffs[head, seq_pos, :] = get_loss_from_ablating_head(layer=layer, head=head, seq_pos=seq_pos, mode=mode)

    imshow(
        loss_diffs,
        facet_col = 0,
        facet_labels = [f"{layer}.{head}" for head in range(model.cfg.n_heads)],
        title = f"Loss from mean ablating the {'output' if mode == 'write' else 'input'} of layer-{layer} attention heads",
        y = LABELS_HTML,
        x = LABELS_HTML[-5:-1],
        labels = {"y": "Written-to position" if mode == 'write' else "Read-from position", "x": "Prediction position"},
        height = 600,
        width = 1000,
    )

plot_all_ablation_loss(layer=0, mode="write")
```
""", unsafe_allow_html=True)

    with open(sum_dir / "fig_ablation.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=600)

    st.markdown(
r"""
Let's establish some more notation, before we discuss our findings:

* Each digit `Ci` has an associated **sum** and a **carry**, i.e. their value is `(sum + int(carry == True)) % 10`
* The carry for `Ci` equals `A(i-1) + B(i-1) >= 10`$ (ignoring carrying across more than one digit for now)
* The sum for `Ci` equals `Ai + Bi`

To calculate the value of each digit, the model has to:

* Storing the sum of `i`-digits (i.e. `Ai + Bi`) at `C(i-1)`, for each `i = 0, 1, 2, 3`
* Storing whether `Ci` should be incremented (i.e. whether `A(i-1) + B(i-1) >= 10`) at `C(i-1)` for each `i = 0, 1, 2`

It's easy to imagine how we could calculate the sum: just uniformly attend to the two digits, then have a head in layer 1 process this information and calculate the sum. But how could we calculate the carry? You might guess this takes 2 attention layers, but actually a very clever algorithm can do it in a single layer.

<details>
<summary>Hint</summary>

Consider just the two units digits. 

* What could we deduce if one of them is a 0?
* What could we deduce if one of them is a 9, and the other one is *not* a 0?
* What could we deduce if one of them is a 1, and the other one is *not* a 9?

Can you generalize this?

</details>

<details>
<summary>1-layer algorithm for computing "carry"</summary>

* If one of the digits is a 0, then carry is False.
* If one of the digits is a 9, and the other is *not* 0, then carry is True.
* If one of the digits is a 1, and the other is *not* 9, then carry is False.

We can generalize this into the following hierarchy:

$$
0 > 9 > 1 > 8 > 2 > 7 > 3 > 6 > 4 > 5
$$

and have an attention head perform the following algorithm: ***attend to the first digit in this hierarchy, and predict "carry" if it's 5 or greater, "not carry" if it's 4 or smaller.***

</details>

From eyeballing the attention patterns, it looks like both of these things are happening. There are some attention heads & destination positions which are doing the "sum" thing (attending uniformly to 2 digits), e.g. `=` attending equally to `A0` and `B0` in head `0.1`. There are also some which look like they're doing the "carry" thing (attending to the digit implied by the hierarchy above), e.g. `C1` attending to either `A3` or `B3` in head `1.0`.

We also get a lot of information from the ablation plots above. In particular, we know that no useful information ever gets stored in sequence positions other than `B3, =, C0, C1, C2` by heads in layer 0, so we can focus on just these.

Let's plot the QK circuits for all these heads, so we can draw stronger conclusions about what the important heads & destination positions are doing, and whether they're storing "sum" or "carry".

```python
def plot_all_QK(cache: ActivationCache, layer: int):
    '''
    Plots a set of rows of the QK matrix for each head in a particular layer.
    '''
    posn_str_list = ["B3", "=", "C0", "C1", "C2"]
    posn_list = [LABELS_DICT[posn_str] for posn_str in posn_str_list]

    # First, get the Q-side matrix (for what's in the residual stream). Easiest way to do this is
    # to take mean over the dataset (equals token will always be the same, and for the others I'm
    # averaging over the digits).
    query_side_resid = (cache["embed"] + cache["pos_embed"])[:, posn_list].mean(0)

    # Use this to get the full QK matrix
    W_QK = model.W_Q[layer] @ model.W_K[layer].transpose(-1, -2)
    W_QK_full = query_side_resid @ W_QK @ model.W_E.T

    fig = make_subplots(rows=1, cols=model.cfg.n_heads, subplot_titles=[f"0.{head}" for head in range(3)])

    for head in range(model.cfg.n_heads):
        for posn in posn_list:
            fig.append_trace(
                go.Bar(
                    name=LABELS_HTML[posn],
                    showlegend=(head == 0),
                    x=[f"{i}" for i in range(10)],
                    y=W_QK_full[head, posn - LABELS_DICT["B3"]].tolist(),
                    marker_color=px.colors.qualitative.D3[posn - LABELS_DICT["B3"]],
                ),
                row = 1, col = head + 1,
            )
    fig.update_layout(
        barmode='group',
        template='simple_white',
        height = 600,
        width = 1300,
        title = f"QK circuits for layer {layer}, using token embeddings",
        legend_title_text = "Dest token",
        yaxis_title_text = "Score",
    )
    fig.show()

    W_QK_pos_full = model.W_pos @ W_QK @ model.W_pos.T
    
    imshow(
        W_QK_pos_full[:, posn_list],
        facet_col=0,
        facet_labels=[f"{layer}.{head}" for head in range(3)],
        height=300,
        width=1300,
        y=posn_str_list,
        x=LABELS_STR,
        title = f"QK circuits for layer {layer}, using positional embeddings",
        labels = {"x": "Source posn", "y": "Dest posn"},
    )

plot_all_QK(cache, layer=0)
```
""", unsafe_allow_html=True)

    with open(sum_dir / "fig_qk_bar1.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=600)
    with open(sum_dir / "fig_qk_imshow.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=300)

    st.markdown(
r"""
Now, we're ready to tentatively draw the following conclusions about which heads & sequence positions matter (and why):

* Only heads in layer 0 are calculating & storing the "sum" or "carry" information (doing the QK plot above for layer 1 produces no discernible patterns)
* `0.0` is calculating:
    * Carry information for `C2`, and storing it at `C1` and `B3` (the latter quite weakly)
* `0.1` is calculating:
    * Carry information for `C1`, and storing it at `B3`
    * Sum information for `C0`, `C2` and `C3`, storing it at `=`, `C1`, `C2` respectively
* `0.2` is calculating:
    * Carry information for `C0`, and storing it at `=`
    * Sum information for `C1`, `C2` and `C3`, storing it at `C0`, `C1`, `C2` respectively

Note that there might be some overlap between calculating sum information and carry information in a few of these cases. There also seem to be some attention patterns which act in the opposite direction of the hierarchy - seems likely these are combining additively with the ones that respect the hierarchy, to store the sum information. But overall, this seems like a decent first pass hypothesis for what features the model is storing in the residual stream at layer 0, and how & where it's storing them.

Before we move on to the next section, let's just plot the patterns for the three "carry information" heads & positions, to make the hierarchy a bit easier to see.

```python
def plot_bar_chart(cache: ActivationCache, head_and_posn_list: List[tuple]):

    # First, get the Q-side matrix (for what's in the residual stream). Easiest way to do this is
    # to take mean over the dataset (equals token will always be the same, and for the others I'm
    # averaging over the digits).
    query_side_resid = (cache["embed"] + cache["pos_embed"]).mean(0)

    # Use this to get the full QK matrix
    W_QK = model.W_Q[0] @ model.W_K[0].transpose(-1, -2)
    W_QK_full = query_side_resid @ W_QK @ model.W_E.T

    # Some translation so we can compare the different patterns more easily
    W_QK_full = W_QK_full - W_QK_full.mean(dim=-1, keepdim=True)
    W_QK_full = W_QK_full / W_QK_full.abs().sum(-1, keepdim=True)

    # Turn from string labels to integers
    head_and_posn_list = [(head, LABELS_DICT[posn]) for head, posn in head_and_posn_list]

    # Reorder the QK matrix according to the hierarchy
    hierarchy = [0, 9, 1, 8, 2, 7, 3, 6, 4, 5]
    W_QK_full = W_QK_full[:, :, hierarchy]

    fig = go.Figure([
        go.Bar(
            name=f"(0.{head}, {LABELS_HTML[posn]})",
            x=[str(i) for i in hierarchy],
            y=W_QK_full[head, posn].tolist(),
            marker_color=px.colors.qualitative.D3[posn-LABELS_DICT["B3"]]
        )
        for (head, posn) in head_and_posn_list
    ])
    fig.update_layout(
        legend_title_text="(Attn head, writing posn)",
        bargap=0.4,
        barmode='group',
        template='simple_white',
        height = 600,
        width = 800,
        title = "QK circuits for 'carrying heads' (translated to make the pattern more visible)",
        hovermode = "x unified",
    )
    fig.show()

plot_bar_chart(cache, head_and_posn_list=[(0, "C1"), (1, "B3"), (2, "=")])
```
""", unsafe_allow_html=True)
    with open(sum_dir / "fig_qk_bar2.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=600)
    st.markdown(
r"""
## Singular Value Decomposition

Now that we have an idea what the layer 0 heads might be detecting and how they're detecting it, let's look at how they're representing it. In other words, we'll look at the OV matrices for the different attention heads.

Since we think the dimensionality of the stored information is pretty small (basically just "sum information" and "carry information"), it makes sense to look at the [singular value decomposition](https://www.lesswrong.com/posts/iupCxk3ddiJBAJkts/six-and-a-half-intuitions-for-svd) of the OV matrices. We'll do this below.

*(Note - this was one of several situations where I used ChatGPT to generate code for the visualisations, I feel obligated to mention that it's great at this and imo people still seem to underuse it!)*

```python
W_OV = model.W_V[0] @ model.W_O[0] # [heads d_model d_model_out]
embeddings = model.W_E[:10] # [vocab d_model]
W_OV_embed = embeddings @ W_OV # [heads vocab d_model]
U_ov, S_ov, V_ov = t.svd(W_OV_embed.transpose(-1, -2))

singular_directions = einops.rearrange(utils.to_numpy(V_ov[:, :, :3]), "head vocab sing -> vocab (head sing)")
df = pd.DataFrame(singular_directions, columns = [f"{i},{j}" for i in range(3) for j in range(3)])
df['Labels'] = [str(i) for i in range(10)]

subplot_titles = []
for head in range(model.cfg.n_heads):
    subplot_titles.extend([f"0.{head}<br>Singular {obj}" for obj in ["Vectors (0, 1)", "Vectors (0, 2)", "Vectors (1, 2)", "Values"]])

fig = make_subplots(
    rows=3,
    cols=4,
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
    subplot_titles=subplot_titles
)

for i, head in enumerate(range(3)):
    for j, (dir1, dir2) in enumerate([(0, 1), (0, 2), (1, 2)]):
        fig.add_trace(
            go.Scatter(
                x=df[f'{i},{dir1}'],
                y=df[f'{i},{dir2}'],
                mode='markers+text',
                text=df['Labels'],
            ),
            row=i+1, col=j+1
        )

fig.update_layout(
    height=1000,
    width=1300,
    showlegend=False,
    title_text="SVD of W<sub>E</sub>W<sub>OV</sub> for layer-0 heads",
    margin_t=150,
    title_y=0.95,
    template="simple_white"
).update_traces(
    textposition='middle right',
    marker_size=5
)

for i, head in enumerate(range(3)):
    fig.add_trace(go.Bar(y=utils.to_numpy(S_ov[head])), row=i+1, col=4)

fig.show()
```
""", unsafe_allow_html=True)

    with open(sum_dir / "fig_svd_1.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=1000)

    st.markdown(
r"""
### Conclusion

A lot of these observations reinforce our previous conclusions, but they provide extra information by telling us ***how*** information is stored, not just suggesting ***that*** it is stored.

* Head `0.0` looks like it stores just "carry information", along the first singular value - in other words, a single direction in the residual stream.
* Heads `0.1` and `0.2` both look like they store "carry information" and "sum information" (although `0.1` focuses more on "carry information" and `0.2` more on "sum information").
* The "sum information" is stored in a circular pattern. In the next section, we'll dive deeper into what this circular pattern means.

## Unembedding matrix structure

We've looked at the start of the model. Now, let's jump to the end, and try to figure out how the model is representing the digits in the output sequence before it eventually converts them into logits.

Let's start by taking a look at the unembedding:

```python
imshow(
    model.W_U.T,
    title = "Unembedding matrix",
    height = 300,
    width = 700,
)
```
""", unsafe_allow_html=True)
    with open(sum_dir / "fig_unembed.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=300)
    st.markdown(
r"""
It looks like only 4 dimensions are used to represent the different possible outputs. Or to put it another way, all logits outputs are a linear combination of 4 different vectors. Note that these vectors look approximately sinusoidal over the digits from 0-9 (they have no entries for later dimensions, which makes sense because `=` and `+` are never predicted by the model). This model was trained with **weight decay**, so it makes sense that sparse weights would be encouraged where possible.

Let's return to the singular value decomposition methods we used in the previous section. As it turns out, there are only 2 important directions in the unembedding matrix:

```python
def plot_svd_single(tensor, title=None):

    # Perform SVD
    U_u, S_u, V_u = torch.svd(tensor)

    # Convert the first two singular directions into a Pandas DataFrame
    singular_directions = utils.to_numpy(V_u[:, :2])
    df = pd.DataFrame(singular_directions, columns=['Dir 1', 'Dir 2'])
    df['Labels'] = [str(i) for i in range(10)]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["First two singular directions", "Singular values"])
    fig.add_trace(go.Scatter(x=df['Dir 1'], y=df['Dir 2'], mode='markers+text', text=df['Labels']), row=1, col=1)
    fig.update_traces(textposition='middle right', marker_size=5)
    fig.add_trace(go.Bar(y=utils.to_numpy(S_u)), row=1, col=2)
    fig.update_layout(height=400, width=750, showlegend=False, title_text=title, template="simple_white")
    fig.show()


plot_svd_single(model.W_U[:, :10], title="SVD of W<sub>U</sub>")
```
""", unsafe_allow_html=True)
    with open(sum_dir / "fig_svd_2.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=400)
    
    st.markdown(
r"""
### Conclusion

We can basically write the unembedding matrix as $W_U = \sigma_1 u_1 v_1^T + \sigma_2 u_2 v_2^T$, where $u_1, u_2$ are two orthogonal directions in the residual stream, and $v_1, v_2$ are the corresponding output directions. Ignoring scale factors, this means we can write the important parts of any residual stream vector $x$ in the final layer as:

$$
\begin{aligned}
x &= \cos(\theta) u_1 + \sin(\theta) u_2 \\
logits &= \cos(\theta) v_1 + \sin(\theta) v_2
\end{aligned}
$$

and the model will predict whatever number most closely matches the angle $\theta$ in the plot above.

To verify this is what's going on, we can plot $x \cdot u_1$ against $x \cdot u_2$ for all the model's predictions (color-coded by the true label). We hope to see the points approximately cluster around the unit circle points in the plot above.

```python
def plot_projections_onto_singular_values(
    svd_tensor: Tensor,
    activations: Tensor = cache["resid_post", 1],
    seq_pos: Optional[int] = None,
    title: Optional[str] = None,
    ignore_carry: bool = False,
):
    '''
    If `ignore_carry`, then we color the digit by its digitsum, not by its actual value. In other words, we 
    ignore the carry value when this is True.
    '''
    labels_all = dataset.toks.clone()
    # If we're coloring by sum, replace labels with values of digit sum modulo 10
    if ignore_carry:
        labels_all[:, -4:] = (labels_all[:, :4] + labels_all[:, 5:9]) % 10

    U, S, V = torch.svd(svd_tensor)

    # Convert the first two singular directions into a Pandas DataFrame
    singular_directions = utils.to_numpy(V[:, :2])
    df = pd.DataFrame(singular_directions, columns=['Direction 1', 'Direction 2'])
    df['Labels'] = [str(i) for i in range(10)]

    fig = px.scatter(
        df, x='Direction 1', y='Direction 2', width=700, height=700, title='First two singular directions' if title is None else title, text='Labels'
    ).update_layout(yaxis=dict(scaleanchor="x", scaleratio=1),template='simple_white').update_traces(textposition='middle right')

    if seq_pos is None:
        activations_flattened = einops.rearrange(activations[:, -5:-1], "batch seq d_model -> (batch seq) d_model")
        labels = einops.rearrange(labels_all[:, -4:], "batch seq -> (batch seq)")
    else:
        activations_flattened = activations[:, seq_pos]
        labels = labels_all[:, seq_pos+1]

    activations_projections = einops.einsum(
        activations_flattened, U[:, :2],
        "batch d_model, d_model direction -> direction batch",
    )

    df2 = pd.DataFrame(utils.to_numpy(activations_projections.T), columns=['u1', 'u2'])
    df2['color'] = utils.to_numpy(labels)

    for trace in px.scatter(df2, x='u1', y='u2', color='color').data:
        fig.add_trace(trace)

    fig.show()


plot_projections_onto_singular_values(
    svd_tensor = model.W_U[:, :10],
    activations = cache['resid_post', 1],
    title = "Projections of residual stream onto singular directions of W<sub>U</sub>"
)
```
""", unsafe_allow_html=True)

    with open(sum_dir / "fig_svd_project_1.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=700)
    
    st.markdown(
r"""
### Conclusion

This confirms what we hypothesized - the residual stream at the end of layer 1 has a single degree of freedom, which we can parametrize by the angle $\theta \in [-\pi, \pi]$. We can see how projecting these points onto the directions $u_1, u_2$ and normalizing them will give us the output we expect.

We might guess that something "logit-lens-y" is going on, where after layer 0 the points roughly cluster in the right location, and get sorted based on the carry information by the heads in layer 1. Sadly this turns out not to be the case (see below), but was worth a try! 

## Ablation experiments to test the "carry information" theory

Let's now run a causal experiment to confirm our hypotheses from earlier about the positions which were calculating the "is carried" information.

I'll do this by deleting the "is carried" information (i.e. the result at (`0.0`, `C1`), (`0.1`, `B3`), (`0.2`, `=`) which we think is where the "is carried" information gets stored for `C2`, `C1`, `C0` respectively) and hope that the projections onto the unembedding singular directions now lose the ability to distinguish between carried vs non-carried digits.

```python
CARRY_POSITIONS = [(0, 'C1', 'C2'), (1, 'B3', 'C1'), (2, '=', 'C0')] # each tuple is (layer0_head, posn_str, posn which we think this is the carry for)

for layer0_head, posn_str, posn_predicted_str in CARRY_POSITIONS:

    posn = LABELS_STR.index(posn_str)
    posn_predicted = LABELS_STR.index(posn_predicted_str)

    def hook_fn(result: Float[Tensor, 'batch seq head d_model'], hook: HookPoint):
        result[:, posn, layer0_head] = result[:, posn, layer0_head].mean(0)
        return result

    model.reset_hooks()
    model.add_hook(utils.get_act_name('result', 0), hook_fn)
    patched_logits, patched_cache = model.run_with_cache(dataset.toks)

    # Plot the first two singular directions, at sequence positions which represent the predictions we think get altered here
    plot_projections_onto_singular_values(
        svd_tensor=model.W_U[:, :10],
        activations=patched_cache['resid_post', 1],
        seq_pos=posn_predicted-1,
        title=f"Patching result of 0.{layer0_head} at {posn_str!r} messes up predictions of carry digit for {posn_predicted_str!r}",
        ignore_carry=False,
    )
```
""", unsafe_allow_html=True)

    for i in range(3):
        with open(sum_dir / f"fig_svd_project_3_head{i}.html", 'r') as f: fig1 = f.read()
        st_html(fig1, height=700)
    
    st.markdown(
r"""
### Conclusion

Our hypothesis is definitely confirmed for the `C2` patching. The model can figure out the sum of 2 digits, but it can't figure out whether to carry, so the cluster around the "$n$-direction" contains digits with the correct answers $n$ and $n+1$. I also added the argument `ignore_carry` to the plotting function, which can be set to `True` to just color the points by the digit sum modulo 10 rather than their actual value. Doing this confirms that the points are being projected onto the correct digit according to this value; it's just the carry information that they can't figure out.

The results are also supported for the `C1` and `C0` patching, although it's a lot messier. This could be for one of two reasons: (1) there's also messy logic regarding when a digit gets carried twice, and (2) some of the stuff we ablated might have been "sum information" rather than just "carry information".

## Linear probes

I'm interested in how the model manages to store a representation of the sum of two digits (ignoring the carrying information for now). Does it do this by the end of layer 0, or only by the end of layer 1?

We'll just apply the probe to the outputs of heads `0.1` and `0.2`, because they're the ones calculating the sum information. We'll also just look at the units digit for now (but results are basically the same when you look at each of the four digits).

```python
class LinearProbe(nn.Module):
    '''
    Basic probe class, with a single linear layer. Code generated by ChatGPT.
    '''
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.fc = nn.Linear(in_features=model.cfg.d_model, out_features=output_dim)

    def forward(self, x: Float[Tensor, "batch d_model"]) -> Float[Tensor, "batch n"]:
        return self.fc(x)


def train_probe(
    output_dim: int,
    dataset: TensorDataset,
    batch_size: int = 100,
    epochs = 50,
    weight_decay = 0.005
) -> LinearProbe:
    '''
    Trains the probe using Adam optimizer. `dataset` should contain the activations and labels.
    '''
    t.set_grad_enabled(True)
    
    probe = LinearProbe(output_dim=output_dim).to(device)

    # Training with weight decay, makes sure probe is incentivised to find maximally informative features
    optimizer = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=weight_decay)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    bar = tqdm(range(epochs))
    for epoch in bar:
        for activations, labels in dataloader:
            optimizer.zero_grad()
            logits = probe.forward(activations)
            loss = F.cross_entropy(logits, labels.long())
            loss.backward()
            optimizer.step()
        bar.set_description(f"Loss = {loss:.4f}")

    t.set_grad_enabled(False)
    return probe


# Creating a large dataset, because probes can sometimes take a while to converge
large_dataset = SumDataset(size=30_000, num_digits=4, seed=42).to(device)
_, large_cache = model.run_with_cache(large_dataset.toks)

# Get probe for sum of digits
output_dim = 10
# activations = outputs of attention heads 0.1 & 0.2 at position C2 (these heads attend to A3 and B3)
activations = large_cache["result", 0][:, LABELS_DICT["C2"], [1, 2]].sum(1)
# labels = (A3 + B3) % 10
labels = (large_dataset.toks[:, LABELS_DICT["A3"]] + large_dataset.toks[:, LABELS_DICT["B3"]]) % 10
trainset = TensorDataset(activations, labels)
probe_digitsum = train_probe(output_dim, trainset, epochs=75, batch_size=300)

# Plot results
plot_svd_single(probe_digitsum.fc.weight.T, title="SVD of directions found by probe")
```
""", unsafe_allow_html=True)
    
    with open(sum_dir / "fig_svd_3.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=400)
    st.markdown(
r"""
Interesting - it looks like the sum of digits is clearly represented in a circular way by the end of layer 0! This is in contrast to just the information about the individual digits, which has a much less obviously circular representation (and has a lot more directions with non-zero singular values).

```python
labels = large_dataset.toks[:, LABELS_DICT["A3"]]
trainset = TensorDataset(activations, labels)
probe_digitA = train_probe(output_dim, trainset, epochs=75, batch_size=300)

# Plot results
plot_svd_single(probe_digitA.fc.weight.T, title="SVD of directions found by probe")
```
""", unsafe_allow_html=True)
    with open(sum_dir / "fig_svd_4.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=400)

    st.markdown(
r"""
How have we managed to represent the direction `(A3 + B3) % 10` in the residual stream at the end of layer 0? Neel Nanda's **Grokking Modular Arithmetic** work might offer a clue. We have trig formulas like:

$$
\sin x \cos y + \cos x \sin y = \sin(x + y)
$$

The heads in layer 0 have two degrees of freedom: learning attention patterns, and learning a mapping from embeddings to output. We might have something like:

* The attention patterns from `C2` to the digits `A3, B3` are proportional to the terms $\sin x, \sin y$
* The output vectors (i.e. from the OV matrix) have components of sizes $\cos y, \cos x$ in some particular direction
* So the linear combination of the output vectors (with attention patterns as linear coefficients) is proportional to $\sin x \cos y + \cos x \sin y = \sin(x + y)$ in this direction

We could imagine getting terms proportional to $\cos(x+y)$ in the same way. So this is how a linear combination of the circular representations of the two individual digits could be turned into a representation of the sum of the two digits.

Another piece of evidence that something like this is possible - I trained a 1-layer model on this task and it achieved an accuracy of around 95%, suggesting that the model basically manages to learn the sum of two digits in a single layer (and the accuracy being below 100% is likely due to cases where the model has to carry digits over two positions, although I didn't check this in detail).

## Final Summary

To calculate each digit `Ci`, we require 2 components - the **sum** and the **carry**. The formula for `Ci` is `(sum + int(carry == True)) % 10`, where `sum` is the sum of digits `Ai + Bi`, and `carry` is whether `A(i+1) + B(i+1) >= 10`. (This ignores issues of carrying digits multiple times, which I won't discuss in this solution.)

We calculate the carry by using the hierarchy $0 > 9 > 1 > 8 > ... > 4 > 5$. An attention head in layer 0 will attend to the first number in this hierarchy that it sees, and if that number is $\geq 5$ then that means the digit will be carried. There are also some layer 0 attention heads which store the sum information in certain sequence positions - either by attending uniformly to both digits, or by following the reverse hierarchy so it can additively combine with something that follows the hierarchy.

At the end of layer 0, the sum information is stored in the residual stream as points around a circle traced out by two vectors, parameterized by an angle $\theta$. The carry information is stored in the residual stream as a single direction.

The model manages to store the sum of the two digits modulo 10 in a circular way by the end of layer 0 (although it's not stored in exactly the same way it will be at the end of the model). We might guess the model takes advantage of some trig identities to do this, although I didn't have time to verify this conclusively.

The heads in layer 1 mostly process this information by self-attending. They don't seem as important as heads `0.1` and `0.2` (measured in terms of loss after ablation), and it seems likely they're mainly clearing up some of the representations learned by the layer 0 heads (and dealing with logic about when to carry digits multiple times).

By the end of layer 1, the residual stream is parameterized by a single value: the angle $\theta$. The digits from 0-9 are evenly spaced around the unit circle, and the model's prediction depends on which angle they're closest to.
""", unsafe_allow_html=True)


def section_0_october():

    st.sidebar.markdown(
r"""

## Table of Contents

<ul class="contents">
    <li><a class='contents-el' href='#prerequisites'>Prerequisites</a></li>
    <li><a class='contents-el' href='#difficulty'>Difficulty</a></li>
    <li><a class='contents-el' href='#motivation'>Motivation</a></li>
    <li><a class='contents-el' href='#logistics'>Logistics</a></li>
    <li><a class='contents-el' href='#what-counts-as-a-solution'>What counts as a solution?</a></li>
    <li><a class='contents-el' href='#setup'>Setup</a></li>
    <li><a class='contents-el' href='#task-dataset'>Task & Dataset</a></li>
    <li><a class='contents-el' href='#model'>Model</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(
r"""
# Monthly Algorithmic Challenge (October 2023): Sorted List

### Colab: [problem](https://colab.research.google.com/drive/1IygYxp98JGvMRLNmnEbHjEGUBAxBkLeU)

This post is the fourth in the sequence of monthly mechanistic interpretability challenges. They are designed in the spirit of [Stephen Casper's challenges](https://www.lesswrong.com/posts/KSHqLzQscwJnv44T8/eis-vii-a-challenge-for-mechanists), but with the more specific aim of working well in the context of the rest of the ARENA material, and helping people put into practice all the things they've learned so far.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/sorted-problem.png" width="350">

## Prerequisites

The following ARENA material should be considered essential:

* **[1.1] Transformer from scratch** (sections 1-3)
* **[1.2] Intro to Mech Interp** (sections 1-3)

The following material isn't essential, but is recommended:

* **[1.2] Intro to Mech Interp** (section 4)
* **July's Algorithmic Challenge - writeup** (on the sidebar of this page)
* Previous algorithmic problems in the sequence

## Difficulty

**This problem is probably the easiest in the sequence so far**, so I expect solutions to have fully reverse-engineered it, as well as presenting adversarial examples and explaining how & why they work.**

## Motivation

Neel Nanda's post [200 COP in MI: Interpreting Algorithmic Problems](https://www.lesswrong.com/posts/ejtFsvyhRkMofKAFy/200-cop-in-mi-interpreting-algorithmic-problems) does a good job explaining the motivation behind solving algorithmic problems such as these. I'd strongly recommend reading the whole post, because it also gives some high-level advice for approaching such problems.

The main purpose of these challenges isn't to break new ground in mech interp, rather they're designed to help you practice using & develop better understanding for standard MI tools (e.g. interpreting attention, direct logit attribution), and more generally working with libraries like TransformerLens.

Also, they're hopefully pretty fun, because why shouldn't we have some fun while we're learning?

## Logistics

The solution to this problem will be published on this page in the first few days of November, at the same time as the next problem in the sequence. There will also be an associated LessWrong post.

If you try to interpret this model, you can send your attempt in any of the following formats:

* Colab notebook,
* GitHub repo (e.g. with ipynb or markdown file explaining results),
* Google Doc (with screenshots and explanations),
* or any other sensible format.

You can send your attempt to me (Callum McDougall) via any of the following methods:

* The [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), via a direct message to me
* My personal email: `cal.s.mcdougall@gmail.com`
* LessWrong message ([here](https://www.lesswrong.com/users/themcdouglas) is my user)

**I'll feature the names of everyone who sends me a solution on this website, and also give a shout out to the best solutions.**

Please don't discuss specific things you've found about this model until the challenge is over (although you can discuss general strategies and techniques, and you're also welcome to work in a group if you'd like). The deadline for this problem will be the end of this month, i.e. 31st August.

## What counts as a solution?

Going through the solutions for the previous problems in the sequence (July: Palindromes & August: First Unique Character) as well as the exercises in **[1.4] Balanced Bracket Classifier** should give you a good idea of what I'm looking for. In particular, I'd expect you to:

* Describe a mechanism for how the model solves the task, in the form of the QK and OV circuits of various attention heads (and possibly any other mechanisms the model uses, e.g. the direct path, or nonlinear effects from layernorm),
* Provide evidence for your mechanism, e.g. with tools like attention plots, targeted ablation / patching, or direct logit attribution.
* (Optional) Include additional detail, e.g. identifying the subspaces that the model uses for certain forms of information transmission, or using your understanding of the model's behaviour to construct adversarial examples.

# Setup

```python
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from pathlib import Path

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "monthly_algorithmic_problems" / "october23_sorted_list"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from monthly_algorithmic_problems.october23_sorted_list.dataset import SortedListDataset
from monthly_algorithmic_problems.october23_sorted_list.model import create_model
from plotly_utils import hist, bar, imshow

device = t.device("cuda" if t.cuda.is_available() else "cpu")
```

## Task & Dataset

The problem for this month is interpreting a model which has been trained to sort a list. The model is fed sequences like:

```
[11, 2, 5, 0, 3, 9, SEP, 0, 2, 3, 5, 9, 11]
```

and has been trained to predict each element in the sorted list (in other words, the output at the `SEP` token should be a prediction of `0`, the output at `0` should be a prediction of `2`, etc).

Here is an example of what this dataset looks like:

```python
dataset = SortedListDataset(size=1, list_len=5, max_value=10, seed=42)

print(dataset[0].tolist())
print(dataset.str_toks[0])
```

<div style='font-family:monospace; font-size:15px;'>
[9, 6, 2, 4, 5, 11, 2, 4, 5, 6, 9]<br>
['9', '6', '2', '4', '5', 'SEP', '2', '4', '5', '6', '9']
</div><br>

The relevant files can be found at:

```
chapter1_transformers/
└── exercises/
    └── monthly_algorithmic_problems/
        └── october23_sorted_list/
            ├── model.py               # code to create the model
            ├── dataset.py             # code to define the dataset
            ├── training.py            # code to training the model
            └── training_model.ipynb   # actual training script
```

## Model

The model is attention-only, with 1 layer, and 2 attention heads per layer. It was trained with layernorm, weight decay, and an Adam optimizer with linearly decaying learning rate.

You can load the model in as follows:

```python
filename = section_dir / "sorted_list_model.pt"

model = create_model(
    list_len=10,
    max_value=50,
    seed=0,
    d_model=96,
    d_head=48,
    n_layers=1,
    n_heads=2,
    normalization_type="LN",
    d_mlp=None
)

state_dict = t.load(filename)

state_dict = model.center_writing_weights(t.load(filename))
state_dict = model.center_unembed(state_dict)
state_dict = model.fold_layer_norm(state_dict)
state_dict = model.fold_value_biases(state_dict)
model.load_state_dict(state_dict, strict=False);
```

The code to process the state dictionary is a bit messy, but it's necessary to make sure the model is easy to work with. For instance, if you inspect the model's parameters, you'll see that `model.ln_final.w` is a vector of 1s, and `model.ln_final.b` is a vector of 0s (because the weight and bias have been folded into the unembedding).

```python
print("ln_final weight: ", model.ln_final.w)
print("\nln_final, bias: ", model.ln_final.b)
```

<details>
<summary>Aside - the other weight processing parameters</summary>

Here's some more code to verify that our weights processing worked, in other words:

* The unembedding matrix has mean zero over both its input dimension (`d_model`) and output dimension (`d_vocab`)
* All writing weights (i.e. `b_O`, `W_O`, and both embeddings) have mean zero over their output dimension (`d_model`)
* The value biases `b_V` are zero (because these can just be folded into the output biases `b_O`)

```python
W_U_mean_over_input = einops.reduce(model.W_U, "d_model d_vocab -> d_model", "mean")
t.testing.assert_close(W_U_mean_over_input, t.zeros_like(W_U_mean_over_input))

W_U_mean_over_output = einops.reduce(model.W_U, "d_model d_vocab -> d_vocab", "mean")
t.testing.assert_close(W_U_mean_over_output, t.zeros_like(W_U_mean_over_output))

W_O_mean_over_output = einops.reduce(model.W_O, "layer head d_head d_model -> layer head d_head", "mean")
t.testing.assert_close(W_O_mean_over_output, t.zeros_like(W_O_mean_over_output))

b_O_mean_over_output = einops.reduce(model.b_O, "layer d_model -> layer", "mean")
t.testing.assert_close(b_O_mean_over_output, t.zeros_like(b_O_mean_over_output))

W_E_mean_over_output = einops.reduce(model.W_E, "token d_model -> token", "mean")
t.testing.assert_close(W_E_mean_over_output, t.zeros_like(W_E_mean_over_output))

W_pos_mean_over_output = einops.reduce(model.W_pos, "position d_model -> position", "mean")
t.testing.assert_close(W_pos_mean_over_output, t.zeros_like(W_pos_mean_over_output))

b_V = model.b_V
t.testing.assert_close(b_V, t.zeros_like(b_V))
```

</details>

A demonstration of the model working:

```python
N = 500
dataset = SortedListDataset(size=N, list_len=10, max_value=50, seed=43)

logits, cache = model.run_with_cache(dataset.toks)
logits: t.Tensor = logits[:, dataset.list_len:-1, :]

targets = dataset.toks[:, dataset.list_len+1:]

logprobs = logits.log_softmax(-1) # [batch seq_len vocab_out]
probs = logprobs.softmax(-1)

batch_size, seq_len = dataset.toks.shape
logprobs_correct = eindex(logprobs, targets, "batch seq [batch seq]")
probs_correct = eindex(probs, targets, "batch seq [batch seq]")

avg_cross_entropy_loss = -logprobs_correct.mean().item()

print(f"Average cross entropy loss: {avg_cross_entropy_loss:.3f}")
print(f"Mean probability on correct label: {probs_correct.mean():.3f}")
print(f"Median probability on correct label: {probs_correct.median():.3f}")
print(f"Min probability on correct label: {probs_correct.min():.3f}")
```

<div style='font-family:monospace; font-size:15px;'>
Average cross entropy loss: 0.039<br>
Mean probability on correct label: 0.966<br>
Median probability on correct label: 0.981<br>
Min probability on correct label: 0.001
</div><br>

And a visualisation of its probability output for a single sequence:

```python
def show(dataset: SortedListDataset, batch_idx: int):
    
    logits: Tensor = model(dataset.toks)[:, dataset.list_len:-1, :]
    logprobs = logits.log_softmax(-1) # [batch seq_len vocab_out]
    probs = logprobs.softmax(-1)

    str_targets = dataset.str_toks[batch_idx][dataset.list_len+1: dataset.seq_len]

    imshow(
        probs[batch_idx].T,
        y=dataset.vocab,
        x=[f"{dataset.str_toks[batch_idx][j]}<br><sub>({j})</sub>" for j in range(dataset.list_len+1, dataset.seq_len)],
        labels={"x": "Token", "y": "Vocab"},
        xaxis_tickangle=0,
        title=f"Sample model probabilities:<br>Unsorted = ({','.join(dataset.str_toks[batch_idx][:dataset.list_len])})",
        text=[
            ["〇" if (str_tok == target) else "" for target in str_targets]
            for str_tok in dataset.vocab
        ],
        width=400,
        height=1000,
    )

show(dataset, 0)
```
""", unsafe_allow_html=True)
    
    with open(sorted_list_dir / "fig_demo.html", 'r', encoding='utf-8') as f: fig1 = f.read()
    st_html(fig1, height=1000)

    st.markdown(r"""
Best of luck! 🎈

""", unsafe_allow_html=True)


def section_1_october():

    st.sidebar.markdown(
r"""

## Table of Contents

<ul class="contents">
    <li><a class='contents-el' href='#summary-of-how-the-model-works'>Summary of how the model works</a></li>
    <li><a class='contents-el' href='#attention-patterns'>Attention patterns</a></li>
    <li><a class='contents-el' href='#ov-qk-circuits'>OV & QK circuits</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#whats-with-the-attention-to-zero'>What's with the attention to zero?</a></li>
        <li><a class='contents-el' href='#advexes'>Advexes</a></li>
    </ul></li>
    <li><a class='contents-el' href='#solving-the-d-d-1-d-2-mystery'>Solving the <code>[d, d+1, d+2]</code> mystery</a></li>
</ul></li>""", unsafe_allow_html=True)
    
    st.markdown(
r"""
# Monthly Algorithmic Challenge (October 2023): Solutions

We assume you've run all the setup code from the previous page "[October] Sorted List". Here's all the new setup code you'll need (we've also added a function to plot all sequences in a dataset, not just one).

```python
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
from typing import List, Optional, cast, Union
import torch as t
from torch import Tensor
import einops
from jaxtyping import Float
from pathlib import Path
import circuitsvis as cv
from transformer_lens import ActivationCache
from eindex import eindex
from transformer_lens import utils
from transformer_lens.hook_points import HookPoint

t.set_grad_enabled(False)

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "monthly_algorithmic_problems" / "october23_sorted_list"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from monthly_algorithmic_problems.october23_sorted_list.model import create_model
from monthly_algorithmic_problems.october23_sorted_list.training import train, TrainArgs
from monthly_algorithmic_problems.october23_sorted_list.dataset import SortedListDataset
from plotly_utils import hist, bar, imshow

device = t.device("cuda" if t.cuda.is_available() else "cpu")

def show_multiple(dataset: SortedListDataset):
    '''
    Visualizes the model's predictions (with option to override data from `dataset` with custom `logits`).

    By default it visualises all sequences in the dataset (raise an exception if this is more than 10).
    '''
    batch_indices = list(range(len(dataset)))
    assert len(dataset) <= 10, "Too many sequences to visualize (max 10)"
        
    logits: Tensor = model(dataset.toks)
    logits = logits[:, dataset.list_len:-1, :]
    logprobs = logits.log_softmax(-1) # [batch seq_len vocab_out]
    probs = logprobs.softmax(-1)

    str_targets = [
        dataset.str_toks[batch_idx][dataset.list_len+1: dataset.seq_len]
        for batch_idx in batch_indices
    ]
    text = [
        [
            ["〇" if (str_tok == target) else "" for target in str_target]
            for str_tok in dataset.vocab
        ]
        for str_target in str_targets
    ]

    imshow(
        probs[batch_indices].squeeze().transpose(-1, -2), # [batch vocab seq]
        y=dataset.vocab,
        labels={"x": "Seq pos", "y": "Prediction"},
        xaxis_tickangle=0,
        width=50 + 300 * len(batch_indices),
        height=1000,
        text=text,
        title="Sample model probabilities",
        facet_col=0,
        facet_labels=[
            f"Unsorted list:<br>({','.join(dataset.str_toks[batch_idx][:dataset.list_len])})"
            for batch_idx in batch_indices
        ]
    )
```

## Summary of how the model works

In the second half of the sequence, the attention heads perform the algorithm "attend back to (and copy) the first token which is larger than me. For example, in a sequence like:

```
[7, 5, 12, 3, SEP, 3, 5, 7, 12]
```

we would have the second 3 token attending back to the first 5 token (because it's the first one that's larger than itself), the second 5 attending back to 7, etc. The SEP token just attends to the smallest token.

Some more refinements to this basic idea:

* The two attending heads split responsibilities across the vocabulary. Head 0.0 is the less important head; it deals with values in the range 28-37 (roughly). Head 0.1 deals with most other values.
* Heads actually sometimes attend more to values like `d+2`, `d+3` than to `d+1` (when `d` is the destination token). So why aren't sequnces with `[d, d+1, d+2]` adversarial examples (i.e. making the model incorrectly predict `d+2` after `d`)?
    * Answer - the OV circuit shows that when we attend to source token `s`, we also boost things slightly less thn `s`, and suppress things slightly more than `s`.
    * So imagine we have a sequence `[d, d+1, d+2]`:
        * Attention to `d+1` will boost `d+1` a lot, and suppress `d+2` a bit. 
        * Attention to `d+2` will boost `d+2` a lot, and boost `d+1` a bit.
        * So even if `d+2` gets slightly more attention, `d+1` might end up getting slightly more boosting.
* Sequences with large jumps are adversarial examples (because they're rare in the training data, which was randomly generated from choosing subsets without replacement). 

## Attention patterns

First, let's visualise attention like we usually do:

```python
cv.attention.from_cache(
    cache = cache,
    tokens = dataset.str_toks,
    batch_idx = list(range(10)),
    radioitems = True,
    return_mode = "view",
    batch_labels = ["<code>" + " ".join(s) + "</code>" for s in dataset.str_toks],
    mode = "small",
)
```
""", unsafe_allow_html=True)
    
    with open(sorted_list_dir / "fig_cv.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=625)

    st.markdown(
r"""
Note, we only care about the attention patterns from the second half of the sequence back to earlier values (since it's a 1-layer model, and that's where we're taking predictions from).

Some observations:

* SEP consistently attends to the smallest value.
* Most of the time, token `d` will attend to the smallest token which is strictly larger than `d`, in at least one of the heads.
    * Seems like heads 0.0 and 0.1 split responsibility across the vocabulary: 0.1 deals with most values, 0.0 deals with a small range of values around ~30.
* This strongly suggests that the heads are predicting whatever they pay attention to.
* One slightly confusing result - sometimes token `d` will pay attention more to the value which is 2 positions higher than `d` in the sorted list, rather than 1 position higher (e.g. very first example: `4` attends more to `7` than to `5`). This is particularly common in sequences with 3 numbers very close together.
    * Further investigation (not shown here) suggests that **these are not adversarial examples**, i.e. attending more to `7` than to `5` doesn't stop `5` from being predicted. At this point, I wasn't sure what the reason for this was.

Next steps - confirm anecdotal observations about OV and QK circuits (plus run some basic head ablation experiments).

## Ablating heads

Testing whether head 0.1 matters more (this was my hypothesis, since it seems to cover more of the vocabulary than 0.0). Conclusion - yes.

```python
def get_loss_from_ablating_head(layer: int, head: int):

    def hook_fn(activation: Float[Tensor, "batch seq nheads d"], hook: HookPoint):
        activation_mean: Float[Tensor, "d_model"] = cache[hook.name][:, :, head].mean(0)
        activation[:, :, head] = activation_mean
        return activation
        
    model.reset_hooks()
    logits_orig = model(dataset.toks)
    logprobs_orig = logits_orig.log_softmax(-1)[:, dataset.list_len:-1, :]
    logits_ablated = model.run_with_hooks(dataset.toks, fwd_hooks=[(utils.get_act_name("result", layer), hook_fn)])
    logprobs_ablated = logits_ablated.log_softmax(-1)[:, dataset.list_len:-1, :]

    targets = dataset.toks[:, dataset.list_len+1:]
    logprobs_orig_correct = eindex(logprobs_orig, targets, "batch seq [batch seq]")
    logprobs_ablated_correct = eindex(logprobs_ablated, targets, "batch seq [batch seq]")

    return (logprobs_orig_correct - logprobs_ablated_correct).mean().item()


print("Loss from mean ablating the output of...")
for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        print(f"  ...{layer}.{head} = {get_loss_from_ablating_head(layer, head):.3f}")
```

<div style='font-family:monospace; font-size:15px;'>
Loss from mean ablating the output of...<br>
  ...0.0 = 0.920<br>
  ...0.1 = 4.963
</div><br>

## OV & QK circuits

We expect OV to be a copying circuit, and QK to be an "attend to anything bigger than self" circuit. `SEP` should attend to the smallest values.

```python
W_OV = model.W_V[0] @ model.W_O[0] # [head d_model_in d_model_out]

W_QK = model.W_Q[0] @ model.W_K[0].transpose(-1, -2) # [head d_model_dest d_model_src]

W_OV_full = model.W_E @ W_OV @ model.W_U

W_QK_full = model.W_E @ W_QK @ model.W_E.T

imshow(
    W_OV_full,
    labels = {"x": "Prediction", "y": "Source token"},
    title = "W<sub>OV</sub> for layer 1 (shows that the heads are copying)",
    width = 900,
    height = 500,
    facet_col = 0,
    facet_labels = [f"W<sub>OV</sub> [0.{h0}]" for h0 in range(model.cfg.n_heads)]
)

imshow(
    W_QK_full,
    labels = {"x": "Input token", "y": "Output logit"},
    title = "W<sub>QK</sub> for layer 1 (shows that the heads are attending to next largest thing)",
    width = 900,
    height = 500,
    facet_col = 0,
    facet_labels = [f"W<sub>QK</sub> [0.{h0}]" for h0 in range(model.cfg.n_heads)]
)
```

""", unsafe_allow_html=True)

    with open(sorted_list_dir / "fig_ov.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=500)
    with open(sorted_list_dir / "fig_qk.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=500)

    st.markdown(
r"""
Conclusion - this basically matches the previous hypotheses:

* Strong diagonal pattern for the OV circuits shows that 0.1 is a copying head on most of the vocabulary (everything outside the values in the [28, 37] range), and 0.1 is a copying head on the other values.
* Weak patchy diagonal pattern in QK circuit shows that most tokens attend more to ones which are slightly above them (and also that there are some cases where `d` attends more to `d+2`, `d+3` etc than to `d+1`).

Visualising that last observation in more detail, for the case `d=25`:

```python
def qk_bar(dest_posn: int):
    bar(
        [W_QK_full[0, dest_posn, :], W_QK_full[1, dest_posn, :]], # Head 1.1, attention from token dest_posn to others
        title = f"Attention scores for destination token {dest_posn}",
        width = 900,
        height = 400,
        template = "simple_white",
        barmode = "group",
        names = ["0.0", "0.1"],
        labels = {"variable": "Head", "index": "Source token", "value": "Attention score"},
    )

qk_bar(dest_posn=25)
```
""", unsafe_allow_html=True)
    
    with open(sorted_list_dir / "fig_qk_bar_25.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=420)

    st.markdown(
r"""
The most attended to are actually 28 and 29! We'll address this later, but first let's also explain a slightly simpler but also confusing-seeming result from the heatmap above.

### What's with the attention to zero?

One weird observation in the heatmap it's worth mentioning - some tokens with very high values (i.e. >35) attend a lot to very small tokens, e.g. zero. 

```python
qk_bar(dest_posn=40)
```
""", unsafe_allow_html=True)

    with open(sorted_list_dir / "fig_qk_bar_40.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=420)

    st.markdown(
r"""
Why don't these tokens all attend to zero? 

Answer - plotting the QK circuit with token embeddings on the query side and positional embeddings on the key side shows that **tokens near the end of the sequence have a bias against attending to very small tokens**. Since tokens near the end of the sequence are likely to be precisely these larger values (i.e. >35), it's reasonable to guess that this effect cancels out the previously observed bias towards small tokens.

```python
POSN_LABELS = [str(i) for i in range(dataset.seq_len)]
POSN_LABELS[dataset.list_len] = "SEP"

W_Qpos_Kemb = model.W_pos @ W_QK @ model.W_E.T

imshow(
    W_Qpos_Kemb,
    labels = {"x": "Key token", "y": "Query position"},
    title = "W<sub>QK</sub> for layer 1 (shows that the heads are attending to next largest thing)",
    y = POSN_LABELS,
    width = 950,
    height = 350,
    facet_col = 0,
    facet_labels = [f"W<sub>QK</sub> [0.{h0}]" for h0 in range(model.cfg.n_heads)]
)
```
""", unsafe_allow_html=True)
    
    with open(sorted_list_dir / "fig_qk_2.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=350)
    st.markdown(
r"""
### Advexes

This plot also reveals a lot of potential advexes - for example, `SEP` consistently attends to the smallest value up to around ~30, where this pattern falls off. So if your entire sequence was in the range [30, 50], it's very possible that the model would fail to correctly identify the smallest token. Can you exhibit an example of this?

Another possible advex: if there's a large gap between tokens `x` and `y`, then `x` might attend to itself rather than to `y`. I created a `CustomSortedList` dataclass to confirm this. I also wrote a function `show_multiple` which can show multiple different plots in a batch at once (this was helpful for quickly testing out advexes) - you can see this in the Setup code section.

```python
class CustomSortedListDataset(SortedListDataset):

    def __init__(self, unsorted_lists: List[List[int]], max_value: int):
        '''
        Creates a dataset from the unsorted lists in unsorted_lists.
        '''
        self.size = len(unsorted_lists)
        self.list_len = len(unsorted_lists[0])
        self.seq_len = 2*self.list_len + 1
        self.max_value = max_value

        self.vocab = [str(i) for i in range(max_value+1)] + ["SEP"]

        sep_toks = t.full(size=(self.size, 1), fill_value=self.vocab.index("SEP"))
        unsorted_list = t.tensor(unsorted_lists)
        sorted_list = t.sort(unsorted_list, dim=-1).values
        self.toks = t.concat([unsorted_list, sep_toks, sorted_list], dim=-1)

        self.str_toks = [[self.vocab[i] for i in toks] for toks in self.toks.tolist()]

        
custom_dataset = CustomSortedListDataset(
    unsorted_lists = [
        [0] + list(range(40, 49)),
        [5] + list(range(30, 48, 2)),
    ],
    max_value=50,
)

custom_logits, custom_cache = model.run_with_cache(custom_dataset.toks)

cv.attention.from_cache(
    cache = custom_cache,
    tokens = custom_dataset.str_toks,
    radioitems = True,
    return_mode = "view",
    batch_labels = ["<code>" + " ".join(s) + "</code>" for s in custom_dataset.str_toks],
    mode = "small",
)

show_multiple(custom_dataset)
```

""", unsafe_allow_html=True)
    with open(sorted_list_dir / "fig_custom_cv.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=450)
    with open(sorted_list_dir / "fig_custom.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=1000)

    st.markdown(
r"""
Conclusion - yes, we correctly tricked `x` into self-attending rather than attending to `y` in these cases. The predictions were a bit unexpected, but we can at least see that the model predicts `x` with non-negligible probability (i.e. showing it's incorrectly predicted the token it attends to), and doesn't predict `y` at all.

## Solving the `[d, d+1, d+2]` mystery

At this point, I spent frankly too long trying to figure out how sequences of the form `[d, d+1, d+2]` *weren't* adversarial for this model. Before eventually finding the correct answer, the options I considered were:

* Maybe the less important head does something valuable. For instance, if there's a token where 0.1 boosts `d+1` and `d+2`, maybe head 0.0 suppresses `d+2`.
    * After all, it does seem from the OV circuit plot like head 0.0 is an anti-copying head at the tokens where 0.1 is a copying head.
    * (However, the same cannot be said for the tokens where 0.0 is a copying head, i.e. 0.1 doesn't seem like it's anti-copying here - which made me immediately suspicious of this explanation.)
* The direct path `W_E @ W_U` is responsible for boosting tokens like `d+1` much more than `d+2`.
    * This proves to kinda be true (see plot below), but if this was the main factor then you'd expect `[d, d+1, d+2]` sequences to become advexes once you remove the direct path. I ran an ablation experiment to test this, and it turned out not to be true.

```python
imshow(
    model.W_E @ model.W_U,
    title = "DLA from direct path",
    labels = {"x": "Prediction", "y": "Input token"},
    height = 500,
    width = 600,
)
```
""", unsafe_allow_html=True)
    
    with open(sorted_list_dir / "fig_direct.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=520)

    st.markdown(
r"""
Finally, I found the actual explanation. As described earlier, **attending to `d+2` will actually slightly boost `d+1`, and attending to `d+1` will slightly suppress `d+2`** (and the same holds true for slightly larger gaps between source tokens). So even if `d+2` is getting a bit more attention, the net effect will be that `d+1` gets boosted more than `d+2`. 

To visualise this, here's a set of 5 examples. Each of them contains sequences with 3 values `x < y < z` close together, which I judged from the QK bar charts earlier would trick the model by having `x` attend to `z` as much as / more than `y`. For each of them, I measured the direct logit attribution to `y` and `z` respectively, coming from the source tokens `y` and `z` respectively. 

I already knew that I would see:

* DLA from `y -> y` large, positive
* DLA from `z -> z` large, positive (often larger than `y -> y`)

And if this hypothesis was correct, then I expected to see:

* DLA from `y -> z` weakly negative
* DLA from `z -> y` weakly positive
* The total DLA to `y` should be larger than the total DLA to `z` (summing over source tokens `y` and `z`)

This is exactly what we see:

```python
custom_dataset = CustomSortedListDataset(
    unsorted_lists = [
        [0, 5, 14, 15, 17, 25, 30, 35, 40, 45],
        [0, 5, 10, 15, 20, 25, 26, 27, 40, 45],
        [0, 5, 10, 15, 20, 25, 30, 31, 32, 45],
        [0, 5, 10, 15, 20, 25, 30, 31, 34, 45],
    ],
    max_value=50,
)
custom_logits, custom_cache = model.run_with_cache(custom_dataset.toks)

# For each sequence, define which head I expect to be the important one, and define
# which tokens are acting as (x, y, z) in each case
head_list = [1, 1, 0, 0]
x_tokens = [14, 25, 30, 30]
y_tokens = [15, 26, 31, 31]
z_tokens = [17, 27, 32, 34]
src_tokens = t.tensor([y_tokens, z_tokens]).T

# Get the positions of (x, y, z) by indexing into the string tokens lists (need to be
# careful that I'm taking them from the correct half of the sequence)
L = custom_dataset.list_len
x_posns = t.tensor([L + toks[L:].index(str(dt)) for dt, toks in zip(x_tokens, custom_dataset.str_toks)])
y_posns = t.tensor([toks.index(str(st)) for st, toks in zip(y_tokens, custom_dataset.str_toks)])
z_posns = t.tensor([toks.index(str(st)) for st, toks in zip(z_tokens, custom_dataset.str_toks)])
src_posns = t.stack([y_posns, z_posns]).T

out = einops.einsum(
    custom_cache["v", 0], model.W_O[0],
    "batch seqK head d_head, head d_head d_model -> batch seqK head d_model",
)
# out = out.sum(2)
out = out[range(4), :, head_list] # [batch seqK d_model]

attn = custom_cache["pattern", 0][range(4), head_list] # [batch seqQ seqK]
result_pre_sum = einops.einsum(
    out, attn,
    "batch seqK d_model, batch seqQ seqK -> batch seqQ seqK d_model",
)

scale = custom_cache["scale"].unsqueeze(-1) # [batch seqQ 1 1]
dla = (result_pre_sum / scale) @ model.W_U # [batch seqQ seqK d_vocab]

# want tensor of shape (4, 2, 2), with dimensions (batch dim, src token = y/z, predicted token = y/z)
dla_from_yz_to_yz = dla[t.arange(4)[:, None, None], x_posns[:, None, None], src_posns[:, None, :], src_tokens[:, :, None]]

fig = imshow(
    dla_from_yz_to_yz,
    facet_col = 0,
    facet_labels = [
        f"Seq #{i}<br>(x, y, z) = ({x}, {y}, {z})"
        for i, (x, y, z) in enumerate(zip(x_list, y_list, z_list))
    ],
    title = "DLA for custom dataset with (x, y, z) close together",
    title_y = 0.95,
    labels = {"y": "Effect on prediction", "x": "Source token"},
    x = ["src = y", "src = z"],
    y = ["pred = y", "pred = z"],
    width = 900,
    height = 400,
    text_auto = ".2f",
)
""", unsafe_allow_html=True)

    with open(sorted_list_dir / "fig_dla.html", 'r') as f: fig1 = f.read()
    st_html(fig1, height=420)


def section_0_november():

    st.sidebar.markdown(
r"""

## Table of Contents

<ul class="contents">
    <li><a class='contents-el' href='#prerequisites'>Prerequisites</a></li>
    <li><a class='contents-el' href='#difficulty'>Difficulty</a></li>
    <li><a class='contents-el' href='#motivation'>Motivation</a></li>
    <li><a class='contents-el' href='#logistics'>Logistics</a></li>
    <li><a class='contents-el' href='#what-counts-as-a-solution'>What counts as a solution?</a></li>
    <li><a class='contents-el' href='#setup'>Setup</a></li>
    <li><a class='contents-el' href='#task-dataset'>Task & Dataset</a></li>
    <li><a class='contents-el' href='#model'>Model</a></li>
</ul></li>""", unsafe_allow_html=True)

    st.markdown(
r"""
# Monthly Algorithmic Challenge (November 2023): Cumulative Sum

### Colab: [problem](https://colab.research.google.com/drive/1kg8HYbwI54vWESjUJ3pcSYGBT_ntxU18)

This post is the fifth in the sequence of monthly mechanistic interpretability challenges. They are designed in the spirit of [Stephen Casper's challenges](https://www.lesswrong.com/posts/KSHqLzQscwJnv44T8/eis-vii-a-challenge-for-mechanists), but with the more specific aim of working well in the context of the rest of the ARENA material, and helping people put into practice all the things they've learned so far.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/cumsum2.png" width="350">

## Prerequisites

The following ARENA material should be considered essential:

* **[1.1] Transformer from scratch** (sections 1-3)
* **[1.2] Intro to Mech Interp** (sections 1-3)

The following material isn't essential, but is recommended:

* **[1.2] Intro to Mech Interp** (section 4)
* **July's Algorithmic Challenge - writeup** (on the sidebar of this page)
* Previous algorithmic problems in the sequence

## Difficulty

**I estimate that this problem is of about average difficulty in the series.** It's probably harder than both the single-layer attention problems, but easier than either of the 2-layer models. However, this problem is unique in introducing MLPs, so your mileage may vary!

## Motivation

Neel Nanda's post [200 COP in MI: Interpreting Algorithmic Problems](https://www.lesswrong.com/posts/ejtFsvyhRkMofKAFy/200-cop-in-mi-interpreting-algorithmic-problems) does a good job explaining the motivation behind solving algorithmic problems such as these. I'd strongly recommend reading the whole post, because it also gives some high-level advice for approaching such problems.

The main purpose of these challenges isn't to break new ground in mech interp, rather they're designed to help you practice using & develop better understanding for standard MI tools (e.g. interpreting attention, direct logit attribution), and more generally working with libraries like TransformerLens.

Also, they're hopefully pretty fun, because why shouldn't we have some fun while we're learning?

## Logistics

The solution to this problem will be published on this page in the first few days of December, at the same time as the next problem in the sequence. There will also be an associated LessWrong post.

If you try to interpret this model, you can send your attempt in any of the following formats:

* Colab notebook,
* GitHub repo (e.g. with ipynb or markdown file explaining results),
* Google Doc (with screenshots and explanations),
* or any other sensible format.

You can send your attempt to me (Callum McDougall) via any of the following methods:

* The [Slack group](https://join.slack.com/t/arena-la82367/shared_invite/zt-1uvoagohe-JUv9xB7Vr143pdx1UBPrzQ), via a direct message to me
* My personal email: `cal.s.mcdougall@gmail.com`
* LessWrong message ([here](https://www.lesswrong.com/users/themcdouglas) is my user)

**I'll feature the names of everyone who sends me a solution on this website, and also give a shout out to the best solutions.**

Please don't discuss specific things you've found about this model until the challenge is over (although you can discuss general strategies and techniques, and you're also welcome to work in a group if you'd like). The deadline for this problem will be the end of this month, i.e. 31st August.

## What counts as a solution?

Going through the solutions for the previous problems in the sequence (July: Palindromes & August: First Unique Character) as well as the exercises in **[1.4] Balanced Bracket Classifier** should give you a good idea of what I'm looking for. In particular, I'd expect you to:

* Describe a mechanism for how the model solves the task, in the form of the QK and OV circuits of various attention heads (and possibly any other mechanisms the model uses, e.g. the direct path, or nonlinear effects from layernorm),
* Provide evidence for your mechanism, e.g. with tools like attention plots, targeted ablation / patching, or direct logit attribution.
* (Optional) Include additional detail, e.g. identifying the subspaces that the model uses for certain forms of information transmission, or using your understanding of the model's behaviour to construct adversarial examples.

## Setup

```python
import os
import sys
import torch as t
from pathlib import Path
from eindex import eindex

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "monthly_algorithmic_problems" / "november23_cumsum"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from monthly_algorithmic_problems.november23_cumsum.dataset import CumsumDataset
from monthly_algorithmic_problems.november23_cumsum.model import create_model
from plotly_utils import hist, bar, imshow

device = t.device("cuda" if t.cuda.is_available() else "cpu")
```

## Task & Dataset

The problem for this month is interpreting a model which has been trained to classify the cumulative sum of a sequence.

The model is fed sequences of integers, and is trained to classify the cumulative sum at a given sequence position. There are 3 possible classifications:

* 0 (if the cumsum is negative),
* 1 (if the cumsum is zero),
* 2 (if the cumsum is positive).

Here is an example (and also a demonstration of all the important attributes of the dataset class you'll be using):

```python
dataset = CumsumDataset(size=1, seq_len=6, max_value=3, seed=40)

print(dataset[0]) # same as (dataset.toks[0], dataset.labels[0])

print(", ".join(dataset.str_toks[0])) # inputs to the model

print(", ".join(dataset.str_labels[0])) # whether the cumsum of inputs is strictly positive
```

<div style='font-family:monospace; font-size:15px;'>
(tensor([ 0,  1, -3, -3, -2,  3]), tensor([1, 2, 0, 0, 0, 0]))<br>
+0, +1, -3, -3, -2, +3<br>
zero, pos, neg, neg, neg, neg
</div><br>

The relevant files can be found at:

```
chapter1_transformers/
└── exercises/
    └── monthly_algorithmic_problems/
        └── november23_cumsum/
            ├── model.py               # code to create the model
            ├── dataset.py             # code to define the dataset
            ├── training.py            # code to training the model
            └── training_model.ipynb   # actual training script
```

## Model

The model is **not attention only**. It has one attention layer with a single head, and one MLP layer. It does *not* have layernorm at the end of the model. It was trained with weight decay, and an Adam optimizer with linearly decaying learning rate.

You can load the model in as follows. Note that this code is different to previous months, because we've removed the layernorm folding.

```python
filename = section_dir / "cumsum_model.pt"

model = create_model(
    max_value=5,
    seq_len=20,
    seed=0,
    d_model=24,
    d_head=12,
    n_layers=1,
    n_heads=1,
    normalization_type=None,
    d_mlp=8,
)

state_dict = t.load(filename)

state_dict = model.center_writing_weights(t.load(filename))
state_dict = model.center_unembed(state_dict)
state_dict = model.fold_value_biases(state_dict)
model.load_state_dict(state_dict, strict=False);
```

A demonstration of the model working:

```python
N = 1000
dataset = CumsumDataset(size=1000, max_value=5, seq_len=20, seed=42).to(device)

logits, cache = model.run_with_cache(dataset.toks)

logprobs = logits.log_softmax(-1) # [batch seq_len vocab_out]
probs = logprobs.softmax(-1)

batch_size, seq_len = dataset.toks.shape
logprobs_correct = eindex(logprobs, dataset.labels, "batch seq [batch seq]")
probs_correct = eindex(probs, dataset.labels, "batch seq [batch seq]")

print(f"Average cross entropy loss: {-logprobs_correct.mean().item():.3f}")
print(f"Mean probability on correct label: {probs_correct.mean():.3f}")
print(f"Median probability on correct label: {probs_correct.median():.3f}")
print(f"Min probability on correct label: {probs_correct.min():.3f}")
```

<div style='font-family:monospace; font-size:15px;'>
Average cross entropy loss: 0.073<br>
Mean probability on correct label: 0.938<br>
Median probability on correct label: 0.999<br>
Min probability on correct label: 0.579
</div><br>

And a visualisation of its probability output for a single sequence:

```python
def show(dataset: SortedListDataset, batch_idx: int):
    
    logits: Tensor = model(dataset.toks)[:, dataset.list_len:-1, :]
    logprobs = logits.log_softmax(-1) # [batch seq_len vocab_out]
    probs = logprobs.softmax(-1)

    str_targets = dataset.str_toks[batch_idx][dataset.list_len+1: dataset.seq_len]

    imshow(
        probs[batch_idx].T,
        y=dataset.vocab,
        x=[f"{dataset.str_toks[batch_idx][j]}<br><sub>({j})</sub>" for j in range(dataset.list_len+1, dataset.seq_len)],
        labels={"x": "Token", "y": "Vocab"},
        xaxis_tickangle=0,
        title=f"Sample model probabilities:<br>Unsorted = ({','.join(dataset.str_toks[batch_idx][:dataset.list_len])})",
        text=[
            ["〇" if (str_tok == target) else "" for target in str_targets]
            for str_tok in dataset.vocab
        ],
        width=400,
        height=1000,
    )

show(dataset, 0)
```
""", unsafe_allow_html=True)
    
    with open(cumsum_dir / "fig_demo.html", 'r', encoding='utf-8') as f: fig1 = f.read()
    st_html(fig1, height=350)

    st.markdown(r"""
Note, it was trained with a lot of weight decay, which is what makes its probabilities sometimes far from 100% (even if accuracy is basically 100%).

Best of luck! 🎈
""", unsafe_allow_html=True)




#     st.markdown(
# r"""

# """, unsafe_allow_html=True)

# with open(sum_dir / "fig_attn.html", 'r') as f: fig1 = f.read()
# st_html(fig1, height=620)

func_page_list = [
    (section_0_november, "[November] Cumulative Sum"),
    (section_1_october, "[October] Solutions"),
    (section_0_october, "[October] Sorted List"),
    (section_0_september, "[September] Sum Of Two Numbers"),
    (section_1_september, "[September] Solutions"),
    (section_0_august, "[August] First Unique Token"),
    (section_1_august, "[August] Solutions"),
    (section_0_july, "[July] Palindromes"),
    (section_1_july, "[July] Solutions"),
]

func_list = [func for func, page in func_page_list]
page_list = [page for func, page in func_page_list]

page_dict = dict(zip(page_list, range(len(page_list))))

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    idx = page_dict[radio]
    func = func_list[idx]
    func()

page()


streamlit_analytics.stop_tracking(
    unsafe_password=st.secrets["analytics_password"],
)