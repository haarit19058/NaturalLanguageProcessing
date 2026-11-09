# SELF-RAG, from Zero: A Self-Contained, Math-First Study Guide

> **Paper:** *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*
> Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi (UW, AI2, IBM). arXiv:2310.11511, published at ICLR 2024.
> **Files in this folder:** `SelfRAG_2310.11511.pdf` (original), `paper_markdown/SelfRAG_2310.11511.md` (marker conversion), `eval_rubric.md` (CS613 rubric).

This guide assumes you know undergraduate probability, linear algebra and basic deep learning (softmax, cross-entropy, gradient descent). It does **not** assume you know anything about retrieval, RAG, instruction tuning or RLHF; those are built up in Part I.

---

## 0. How to use this guide

| Part | What you get | Rubric criterion it serves |
|---|---|---|
| **I. Prerequisites** (§1–§6) | LMs as probability models, retrieval, classic RAG, and why vanilla RAG fails | Problem & Motivation |
| **II. The paper** (§7–§12) | Problem, core idea, formal model, training math, inference math, worked numeric examples | Core Contribution, Technical Understanding |
| **III. Experiments** (§13–§15) | Tasks, metrics *with formulas*, baselines, main results, ablations, how to read them | Experiments & Results |
| **IV. Critique & impact** (§16–§17) | Strengths, limitations, hidden assumptions, follow-up work | Critical Analysis, Impact |
| **V. Exam prep** (§18–§21) | Q&A bank, self-check exercises with solutions, slide plan, glossary | Communication & Q&A |

**Suggested path for a first-timer.** Read §1–§6 slowly (about 2 hours) and redo every derivation on paper. Then read the paper's §1–§3 next to Part II of this guide. Finish with the exercises in §19, and don't look at the solutions until you have tried them.

### Notation used throughout

| Symbol | Meaning |
|---|---|
| $x$ | input / prompt / instruction |
| $y = [y_1,\dots,y_T]$ | output, split into $T$ **segments** (in this paper, a segment is one **sentence**) |
| $y_{<t}$ | all segments before segment $t$ |
| $w_1,\dots,w_n$ | tokens (sub-word units) |
| $V$ | the LM's original vocabulary |
| $\mathcal{R}$ | the *retriever* (a search model) |
| $\mathcal{C}$ | the *critic* LM (used only during training-data creation) |
| $\mathcal{M}$ | the *generator* LM (the final Self-RAG model) |
| $d$, $D$ | one retrieved passage, a set of retrieved passages |
| $K$ | number of passages retrieved per step |
| $B$ | beam width (number of partial outputs kept) |
| $r$ | a reflection token |
| $G$ | a critique-token *group* (e.g. ISREL) |
| $s_t^G$ | normalized score of group $G$ at segment $t$ |
| $w^G$ | user-chosen weight for group $G$ at inference |
| $\delta$ | retrieval threshold |

---

# PART I: PREREQUISITES

## 1. A language model is a probability distribution over sequences

### 1.1 Autoregressive factorization

A (decoder-only) language model such as Llama 2 defines a probability for every token sequence $w_{1:n}$ using the **chain rule of probability**, with no approximation at all:

$$
p_\theta(w_{1:n}) \;=\; \prod_{i=1}^{n} p_\theta(w_i \mid w_{<i}).
$$

At each position the network (a Transformer) reads the prefix $w_{<i}$ and produces a hidden vector $h_i \in \mathbb{R}^{h}$. A linear "LM head" $W \in \mathbb{R}^{|V|\times h}$ turns it into **logits** $z_i = W h_i \in \mathbb{R}^{|V|}$, and a softmax gives a distribution over the vocabulary:

$$
p_\theta(w_i = v \mid w_{<i}) \;=\; \frac{\exp(z_{i,v})}{\sum_{u\in V}\exp(z_{i,u})}.
$$

**Conditional generation** just conditions on the prompt: $p_\theta(y \mid x) = \prod_i p_\theta(y_i \mid x, y_{<i})$. The prompt $x$ is simply the first part of the sequence.

### 1.2 Training = maximum likelihood = minimizing cross-entropy

Given a dataset of sequences, we maximize the log-likelihood, which is the same as minimizing the **negative log-likelihood (NLL)**, also called token-level cross-entropy:

$$
\mathcal{L}(\theta) \;=\; -\,\mathbb{E}_{(x,y)\sim\mathcal{D}} \sum_{i=1}^{|y|} \log p_\theta(y_i \mid x, y_{<i}).
$$

Two details matter for Self-RAG:

1. **Teacher forcing.** During training the model always conditions on the *true* prefix $y_{<i}$, not on its own guesses. So one forward pass computes the loss for all positions in parallel.
2. **Loss masking.** You can choose *which positions contribute to the loss*. Write a binary mask $m_i\in\{0,1\}$:
   $$\mathcal{L} = -\sum_i m_i \log p_\theta(w_i\mid w_{<i}).$$
   Instruction tuning usually sets $m_i=0$ on prompt tokens (the model should read the prompt, not learn to produce it). **Self-RAG also sets $m_i=0$ on retrieved passage tokens** (§10.3).

### 1.3 Why the softmax over a *subset* matters (key trick in Self-RAG)

Suppose we care only about a subset of tokens $G\subset V$, for example $G=\{\texttt{Yes},\texttt{No}\}$. The probability of token $r\in G$ **given that the next token is in $G$** is

$$
p(r \mid \text{next}\in G) \;=\; \frac{p(r)}{\sum_{r'\in G} p(r')} \;=\; \frac{\exp(z_r)}{\sum_{r'\in G}\exp(z_{r'})}.
$$

This is a softmax restricted to the logits of $G$; the global normalizer cancels. **Every Self-RAG score (§11) is exactly this quantity.** Remember this identity.

### 1.4 Adding new tokens to a vocabulary

Adding tokens $R$ (here, reflection tokens) means growing two matrices:

$$
E \in \mathbb{R}^{|V|\times h} \;\to\; E' \in \mathbb{R}^{(|V|+|R|)\times h}, \qquad W \in \mathbb{R}^{|V|\times h} \;\to\; W' \in \mathbb{R}^{(|V|+|R|)\times h}.
$$

$E$ is the input embedding table and $W$ the output head. The new rows are learned by fine-tuning. After that, *the model emits a reflection token exactly like it emits the word "the"*: through the same softmax. There is no extra classifier head, so **"critique" becomes next-token prediction.** In the released checkpoints these tokens are strings such as `[Retrieval]`, `[No Retrieval]`, `[Continue to Use Evidence]`, `[Relevant]`, `[Irrelevant]`, `[Fully supported]`, `[Partially supported]`, `[No support / Contradictory]`, `[Utility:1]`…`[Utility:5]`, plus `<paragraph>`/`</paragraph>` to wrap passages.

### 1.5 Pre-training vs instruction tuning vs RLHF (one paragraph each)

- **Pre-training:** NLL on trillions of web tokens. Result: a model that continues text (e.g. *Llama2-7B*).
- **Instruction tuning (SFT):** NLL on (instruction, good response) pairs. Result: a model that follows instructions (e.g. *Alpaca*). **Self-RAG's generator is trained with plain SFT**, on a special corpus.
- **RLHF:** train a *reward model* $r_\phi(x,y)$ from human preferences, then optimize the policy with PPO to maximize $\mathbb{E}[r_\phi] - \beta\,\mathrm{KL}(\pi_\theta\Vert\pi_{\text{ref}})$. This is expensive: several models live in memory and samples are drawn on-policy. (e.g. *ChatGPT*, *Llama2-chat*.) Self-RAG takes the idea of *learned critique* from RLHF but **avoids RL entirely** (§10.4).

---

## 2. Parametric vs. non-parametric knowledge, and why LMs hallucinate

- **Parametric memory:** facts stored implicitly in the weights $\theta$. It is fixed after training, has no source, and can't be updated without retraining.
- **Non-parametric memory:** an external collection of documents (e.g. all of Wikipedia, split into ~100-word passages). It can be updated, inspected and cited.

**Hallucination** means fluent, confident but false output. A useful mental model: the LM samples from $p_\theta(y\mid x)$, which is trained to be *plausible*, not *true*. For rare facts the model has seen too few examples, so probability mass spreads over plausible-sounding wrong answers.

**Long-tail evidence (PopQA, Mallen et al. 2023, same group as Self-RAG):** LM accuracy on factual questions falls sharply with how *popular* the entity is. Retrieval helps most exactly on unpopular entities. That's why the paper evaluates on the **long-tail subset of PopQA** (1,399 questions about entities with <100 monthly Wikipedia views). Look at Table 2: Llama2-13B gets **14.7%** without retrieval and **45.7%** with it.

---

## 3. Retrieval: how a search model finds passages

A retriever $\mathcal{R}$ maps a query $q$ and a corpus $\{d_1,\dots,d_N\}$ to the top-$K$ passages by a relevance score $\text{sim}(q,d)$.

### 3.1 Sparse retrieval (BM25), for intuition

$$
\text{BM25}(q,d)=\sum_{t\in q}\text{IDF}(t)\cdot\frac{f(t,d)\,(k_1+1)}{f(t,d)+k_1\left(1-b+b\frac{|d|}{\text{avgdl}}\right)}.
$$

Here $f(t,d)$ is the term frequency, $\text{IDF}(t)=\log\frac{N-n_t+0.5}{n_t+0.5}$ (rarer terms count more), and $k_1\approx1.2$, $b\approx0.75$ control saturation and length normalization. It is pure word overlap, so it misses synonyms.

### 3.2 Dense retrieval (what Self-RAG uses: Contriever-MS MARCO)

A **bi-encoder** Transformer $E(\cdot)$ embeds query and passage separately into $\mathbb{R}^{768}$, and relevance is a dot product:

$$
\text{sim}(q,d) = \langle E(q),\,E(d)\rangle .
$$

Because passages are embedded *offline* once, search becomes **Maximum Inner Product Search (MIPS)** over $N\approx 2\times10^7$ vectors, which approximate-nearest-neighbour libraries (FAISS) do in milliseconds.

**How it's trained: the contrastive (InfoNCE) loss.** For query $q$, a positive passage $d^+$ and negatives $d^-_1,\dots,d^-_m$:

$$
\mathcal{L}_{\text{InfoNCE}} = -\log\frac{\exp(\text{sim}(q,d^+)/\tau)}{\exp(\text{sim}(q,d^+)/\tau)+\sum_{j=1}^m\exp(\text{sim}(q,d_j^-)/\tau)}.
$$

This is cross-entropy for "pick the positive out of $m+1$ candidates". **Contriever** pre-trains this way *without labels* (positives = two random crops of the same document), then is fine-tuned on MS MARCO (real search-engine query→passage pairs).

> **Key fact for critique:** Self-RAG **does not train the retriever.** $\mathcal{R}$ is frozen and off-the-shelf. All the "intelligence about relevance" comes from the LM judging passages *after* retrieval.

---

## 4. Retrieval-Augmented Generation (RAG): the classic formulations

### 4.1 Lewis et al. (2020) "RAG": retrieval as a latent variable

Treat the retrieved passage $z$ as a hidden variable and **marginalize** over it:

$$
p(y\mid x) \;\approx\; \sum_{z\in\text{top-}K(x)} \underbrace{p_\eta(z\mid x)}_{\text{retriever}}\;\underbrace{\prod_i p_\theta(y_i\mid x,z,y_{<i})}_{\text{generator}}, \qquad p_\eta(z\mid x)\propto\exp\langle E(z),E(x)\rangle.
$$

This is **RAG-Sequence**: one passage explains the whole answer. **RAG-Token** marginalizes per token: $\prod_i\sum_z p_\eta(z\mid x)\,p_\theta(y_i\mid x,z,y_{<i})$. Both retriever and generator can be trained end-to-end, because the sum is differentiable with respect to $\eta$.

### 4.2 "In-context RALM" (Ram et al. 2023): the version used in practice today

Retrieve top-$K$ and **concatenate them into the prompt**:

$$
y \sim p_\theta\big(\,\cdot \mid [d_1;\dots;d_K;\,x]\big).
$$

No training is needed and it works with any LM, including ChatGPT. This is what the paper's "**Ret-ChatGPT**", "**Ret-Llama2-chat**", "**Alpaca + retrieval**" baselines do.

### 4.3 Contrast with Self-RAG (preview)

| | Lewis-RAG | In-context RAG | **Self-RAG** |
|---|---|---|---|
| When to retrieve | always, once | always, once | **model decides, per segment** |
| How passages combine | soft mixture (marginalize) | concatenate all $K$ | **one passage per candidate, then select** |
| Relevance check | retriever score only | none | **LM predicts ISREL** |
| Grounding check | none | none | **LM predicts ISSUP** |
| Citations | no | prompt-dependent | **built-in (each segment tied to one $d$)** |
| Controllable at test time | no | no | **yes (weights $w^G$, threshold $\delta$)** |

---

## 5. Why vanilla RAG is not enough: the problem Self-RAG attacks

Write the in-context RAG decision rule as "always retrieve $K$, always use all $K$". The paper lists three failure modes (Sec. 1):

1. **Indiscriminate retrieval hurts versatility.** "Write an essay about your best summer vacation" needs no facts. Forcing in Wikipedia passages wastes compute and can derail creative or reasoning tasks. Formally, retrieval is useful only if
   $\mathbb{E}[\text{quality}\mid\text{retrieve}] > \mathbb{E}[\text{quality}\mid\text{no retrieve}]$, and that depends on $x$ *and* on what has been generated so far, $y_{<t}$.
2. **Irrelevant passages distract.** Shi et al. (2023) show that LMs are "easily distracted by irrelevant context". If $d$ is off-topic, $p_\theta(y\mid x,d)$ can be *worse* than $p_\theta(y\mid x)$.
3. **No guarantee of grounding / attribution.** Even with relevant passages, nothing forces the output to follow them. The model was never trained to check "is my sentence entailed by the evidence?". Gao et al. (2023, ALCE) found that most open models produce poor citations (Table 2: Llama2/Alpaca + retrieval reach only **2–8% citation precision/recall**).

**Prior partial fixes and their gaps** (Sec. 2):
- *Adaptive retrieval* (FLARE / Active RAG, Jiang et al. 2023): retrieve when token confidence is low. It works only by prompting a proprietary LLM and has no notion of grounding.
- *Toolformer* (Schick et al. 2023): an LM learns to emit API calls. It focuses on *calling* tools, not *judging* their output.
- *SAIL* (Luo et al. 2023): instruction-tune with retrieved passages always prepended. Still indiscriminate.
- *Filter/compress passages with extra models* (Yoran et al.: NLI filter; RECOMP: summarizer). These need extra models at inference and judge only relevance.
- *RLHF with critics* (Ouyang et al., Wu et al. fine-grained RLHF): expensive PPO, and the critic shapes training only, not test-time behaviour.
- *Self-refine / Chain-of-Verification*: iterative prompting, which costs many LM calls at inference.

---

## 6. Two more tools you need: beam search and knowledge distillation

### 6.1 Beam search

Greedy decoding picks $\arg\max$ each step and can get trapped. **Beam search** keeps the $B$ best partial hypotheses by cumulative score; at each step it expands each one, scores all children, and keeps the top $B$. Standard beam search works over *tokens* with score $\sum\log p$. **Self-RAG does beam search over *segments* (sentences)** with a *custom* score that includes critique (§11.4).

### 6.2 Knowledge distillation

Use a large "teacher" to label data, then train a small "student" on those labels with ordinary cross-entropy. Self-RAG distills twice:

$$
\underbrace{\text{GPT-4}}_{\text{teacher}} \xrightarrow{\ \text{labels } \sim 4\text{k–}20\text{k per type}\ } \underbrace{\mathcal{C}\ (\text{Llama2-7B critic})}_{\text{student 1 / teacher 2}} \xrightarrow{\ \text{labels } 150\text{k examples}\ } \underbrace{\mathcal{M}\ (\text{generator})}_{\text{student 2}}
$$

Keep this chain in your head; it explains most of the training section.

---

# PART II: THE PAPER

## 7. Problem & motivation (one-slide version)

> **Problem.** LLMs hallucinate. RAG reduces this but (a) retrieves *always*, even when useless, (b) feeds in passages *regardless of relevance*, and (c) never checks whether the output is *supported* by them.
>
> **Why it matters.** Factual errors in deployed assistants spread misinformation and give bad advice. Users also need **verifiable** answers (citations) they can check.
>
> **Motivation.** Could one LM learn to *decide when to look things up*, *judge what it found*, and *grade its own output*, cheaply (no RLHF) and controllably (adjustable at test time)?

---

## 8. The core idea: reflection tokens

**One sentence:** *Extend the LM's vocabulary with special "reflection tokens" and train the LM, with ordinary next-token prediction, to emit them interleaved with its answer. These tokens (i) trigger retrieval on demand and (ii) critique the retrieved passage and its own output. At test time, use their probabilities to steer decoding.*

### 8.1 The four token types (paper Table 1)

| Type | Conditions on | Possible values (★ = most desirable) | Question it answers |
|---|---|---|---|
| **Retrieve** | $x$ (and $y_{<t}$) | `Yes`, `No`, `Continue` | Should I fetch evidence for the *next* segment? (`Continue` = reuse the passage I already have) |
| **ISREL** (is relevant) | $x, d$ | ★`Relevant`, `Irrelevant` | Does passage $d$ help answer $x$? |
| **ISSUP** (is supported) | $x, d, y_t$ | ★`Fully supported`, `Partially supported`, `No support / Contradictory` | Is every verification-worthy claim in $y_t$ backed by $d$? |
| **ISUSE** (is useful) | $x, y$ | ★`5`, `4`, `3`, `2`, `1` | Is $y$ a helpful response to $x$, *regardless of factuality*? ("perceived utility") |

The last three are called **critique tokens**. Retrieve, ISREL and ISSUP are emitted **per segment**; ISUSE is emitted **once, at the end**.

### 8.2 What a training target actually looks like (paper Fig. 2 / Table 4)

```
Input:  when did walking dead season 7 come out
Output: [Retrieve=Yes] <p> The Walking Dead (season 7) ... premiered on October 23, 2016 ... </p>
        [ISREL=Relevant] October 23, 2016 [ISSUP=Fully Supported] [ISUSE=5]

Input:  Synthesize a poem based on this prompt: "The sun is setting"
Output: [Retrieve=No] The stars begin to twinkle bright ... [Retrieve=No] They flicker like a silent dance ... [ISUSE=5]
```

The poem never retrieves; the factual question does. **The model learns this policy from data.**

### 8.3 What is novel (be precise in Q&A)

1. **Retrieval on demand, decided by the LM itself, at segment granularity.** Not once per query, and not via a hand-coded confidence rule.
2. **Self-critique as vocabulary.** Relevance, support and utility judgments are *generated tokens*, so they come for free in the same forward pass. No separate reward model is needed at inference.
3. **Critique learned offline, without RL.** A critic labels the corpus once; the generator is trained with plain SFT.
4. **Controllable decoding.** Critique-token probabilities define a tunable score (weights $w^G$, threshold $\delta$), so one trained model can be made "more cautious" or "more fluent" **without retraining**.
5. **Fine-grained citations.** Each segment is generated from exactly one passage and comes with a self-assessed support label.

---

## 9. Formal model

### 9.1 Segments

Input $x$; output $y=[y_1,\dots,y_T]$, where each $y_t$ is a token sequence (a sentence, split with spaCy). Each $y_t$ may contain normal tokens from $V$ **and** reflection tokens from $R$. The model's vocabulary is $V\cup R$.

### 9.2 The generative story of one segment

For segment $t$, given $(x, y_{<t})$:

$$
\begin{aligned}
&r^{\text{ret}}_t \sim p_{\mathcal{M}}(\,\cdot\mid x, y_{<t}) && \text{(Retrieve token)}\\
&\textbf{if } r^{\text{ret}}_t=\texttt{Yes}:\quad d_t \leftarrow \mathcal{R}(x, y_{t-1}) && \text{(external call, not modeled)}\\
&\qquad r^{\text{rel}}_t \sim p_{\mathcal{M}}(\,\cdot\mid x, d_t, y_{<t}) && \text{(ISREL)}\\
&\qquad y_t \sim p_{\mathcal{M}}(\,\cdot\mid x, d_t, r^{\text{rel}}_t, y_{<t}) && \text{(the actual sentence)}\\
&\qquad r^{\text{sup}}_t \sim p_{\mathcal{M}}(\,\cdot\mid x, d_t, y_t, \dots) && \text{(ISSUP)}\\
&\textbf{else}:\quad y_t \sim p_{\mathcal{M}}(\,\cdot\mid x, y_{<t})
\end{aligned}
$$

and after the last segment, $r^{\text{use}} \sim p_{\mathcal{M}}(\cdot\mid x, y)$.

Write the whole interleaved string as $\tilde y$. Because everything is in one token stream, the model's probability of the *model-produced* part factorizes by the chain rule:

$$
p_{\mathcal{M}}(\tilde y\setminus D \mid x)=\prod_{t=1}^{T}\Big[p(r^{\text{ret}}_t\mid\cdot)\;\big(p(r^{\text{rel}}_t\mid\cdot)\,p(y_t\mid\cdot)\,p(r^{\text{sup}}_t\mid\cdot)\big)^{\mathbb{1}[\text{ret}_t]}\big(p(y_t\mid\cdot)\big)^{1-\mathbb{1}[\text{ret}_t]}\Big]\;p(r^{\text{use}}\mid\cdot).
$$

Passages $D$ are **inputs supplied by $\mathcal{R}$, not outputs of $\mathcal{M}$**, which is why they are excluded from the product (and masked from the loss, §10.3).

### 9.3 Why this order of tokens? (common Q&A question)

The order is a deliberate design choice, because in an autoregressive model **earlier tokens condition later ones**:

- **Retrieve** comes *before* the segment: you must decide to look up *before* writing.
- **ISREL** comes *after the passage but before the sentence*. The model first commits to "this passage is irrelevant", and that judgment conditions the sentence. It learns to rely less on irrelevant passages (paper App. C.1: when ISREL=Irrelevant, Self-RAG signals the answer may be ungrounded).
- **ISSUP** comes *after* the sentence: it is a post-hoc entailment check of $y_t$ against $d$, so it can't come earlier.
- **ISUSE** comes at the very *end*: utility is a property of the whole answer.

---

## 10. Training (paper §3.2, App. A.2, Algorithms 2 & 3)

Training has **two stages**: first build the critic $\mathcal{C}$, then use $\mathcal{C}$ and $\mathcal{R}$ to annotate a corpus and train the generator $\mathcal{M}$ on it.

```
           ┌──────────── Stage 1: Critic ────────────┐     ┌──────────────── Stage 2: Generator ────────────────┐
 (x,y) ──► GPT-4 + type-specific few-shot prompts ──► D_critic ──► SFT Llama2-7B ──► C
                                                                                  │
 150k (x,y) ──► for each sentence y_t: C says Retrieve? ──► R fetches passages ──► C labels ISREL/ISSUP/ISUSE
                                       ──► interleaved corpus D_gen ──► SFT Llama2-7B/13B (+ new tokens) ──► M
 Inference uses ONLY M and R.  C and GPT-4 are gone.
```

### 10.1 Stage 1a: collecting critic labels from GPT-4

Hand-labelling every sentence × passage is too expensive, and **calling GPT-4 on the full corpus would also be too expensive** (150k examples × several sentences × up to 10 passages × 4 judgment types) and not reproducible. So the authors call GPT-4 only on a **sample**:

- For each token type, sample $\{X^{\text{sample}},Y^{\text{sample}}\}\sim\{X,Y\}$.
- Prompt GPT-4 with a **type-specific instruction + few-shot demonstrations** (full prompts are in App. D, Tables 8–12). Example for Retrieve: *"Given an instruction, make a judgment on whether finding some external documents from the web helps to generate a better response."*
- GPT-4 outputs the token as text: $r\sim p_{\text{GPT-4}}(r\mid I, x, y)$, with temperature 1 and at most 200 output tokens. Outputs in the wrong format are discarded.
- **Collected:** 12,594 (Retrieve), 19,317 (ISREL), 11,181 (ISSUP), 3,831 (ISUSE).
- **Sanity check:** the authors hand-checked 20 per type. Agreement with GPT-4 was 95% for relevance, 95% for retrieval need, 90% for support and 80% for utility (disagreements were mostly 4-vs-5 or 1-vs-2).

### 10.2 Stage 1b: training the critic (paper Eq. 1)

$$
\boxed{\;\max_{\mathcal{C}}\;\mathbb{E}_{((x,y),r)\sim\mathcal{D}_{\text{critic}}}\;\log p_{\mathcal{C}}(r\mid x,y)\;}
$$

**Read it as:** "make the critic assign high probability to the label GPT-4 gave". It is *conditional language modeling*: the input $(x,y)$ (plus $d$ where relevant) is the prompt, and the target is the short label $r$. Equivalently, it minimizes cross-entropy between the critic's distribution over label tokens and the one-hot GPT-4 label: a **classifier implemented as an LM**. $\mathcal{C}$ is initialized from **Llama2-7B**.

**How good is the critic?** Accuracy against held-out GPT-4 labels (App. Table 5): Retrieve **93.8**, ISSUP **93.5**, ISREL **80.2**, ISUSE **73.5** (Llama2-7B). A FLAN-3B critic does worse on most types, so a bigger base helps.

### 10.3 Stage 2a: building the generator corpus (Algorithm 3)

For every one of ~**145,619** (x, y) pairs (instruction data: GPT-4 Alpaca, Stanford Alpaca, FLAN-V2, ShareGPT, OpenAssistant; plus knowledge-intensive data: Wizard of Wikipedia, Natural Questions, FEVER, OpenBookQA, ARC-Easy, ASQA):

1. $\mathcal{C}$ predicts **Retrieve** for $(x,y)$. If **No**: only predict **ISUSE**, append it, done.
2. If **Yes**: retrieve with $(x, y)$ as the query, then split $y$ into sentences.
3. For each sentence $y_t$: $\mathcal{C}$ predicts whether retrieval is needed (given $x$, $y_{<t}$ and the earlier passage: **Yes / No / Continue**). If Yes, retrieve with $(x, y_t)$ as the query.
4. For each retrieved $d$: $\mathcal{C}$ predicts **ISREL**, then **ISSUP**.
5. **Choose which passage to insert:** among passages with ISREL=Relevant and ISSUP ∈ {Fully, Partially}, take the highest-retrieval-score one. If none qualifies, **pick a random one**. This deliberately keeps *negative* examples (irrelevant/unsupported) in the data, so $\mathcal{M}$ learns to emit `Irrelevant`/`No support` honestly.
6. At the end, $\mathcal{C}$ predicts **ISUSE**. Add the augmented sequence to $\mathcal{D}_{\text{gen}}$.

**Important subtlety:** the *output text* $y$ is **not rewritten**. It is the original human/GPT-written answer. The critic only *inserts* reflection tokens and passages around it. So the generator learns "this is what good answers look like, and here is how to annotate them", not new answer content.

### 10.4 Stage 2b: training the generator (paper Eq. 2)

$$
\boxed{\;\max_{\mathcal{M}}\;\mathbb{E}_{(x,y,r)\sim\mathcal{D}_{\text{gen}}}\;\log p_{\mathcal{M}}(y, r\mid x)\;}
$$

Written at the token level, with $\tilde y = (\tilde w_1,\dots,\tilde w_n)$ the interleaved sequence:

$$
\mathcal{L}_{\text{gen}}(\mathcal{M}) = -\sum_{i=1}^{n} m_i\,\log p_{\mathcal{M}}(\tilde w_i\mid x,\tilde w_{<i}),\qquad
m_i=\begin{cases}0 & \tilde w_i \text{ inside } \texttt{<p>}\dots\texttt{</p>} \text{ (retrieved passage)}\\ 1 & \text{otherwise (answer text or reflection token)}\end{cases}
$$

**Differences from the critic loss (Eq. 1):**
- The critic predicts *only* $r$ given $(x,y)$. The generator predicts *both* $y$ and $r$, **interleaved**. It is a single model that writes *and* grades.
- The vocabulary is expanded: $V \to V\cup\{\text{Critique},\text{Retrieve}\}$ tokens (§1.4).

**Why mask the passages?** (a likely exam question)
1. The passages are *not produced by the model*; at inference the retriever inserts them. Training to predict them would spend capacity on **memorizing Wikipedia**.
2. Worse, it would teach the model to **hallucinate passage-like text**, the opposite of grounding.
3. The model still *attends* to passage tokens (they are in the context), so it learns to **use** them without learning to **produce** them.

**Why not RLHF/PPO? A cost comparison**

| | PPO-based RLHF | Self-RAG |
|---|---|---|
| Models in GPU memory during training | policy + reference + reward + value (≈4 LMs) | **1** (the generator) |
| Data | on-policy samples generated *during* training | **static** corpus, labelled once offline |
| Objective | high-variance policy gradient + KL penalty | **cross-entropy** (stable, standard) |
| Change behaviour after training | re-train | **change $w^G$, $\delta$ at inference** |

Reported compute: 4×A100-80GB, 3 epochs, batch size 128, peak LR $2\times10^{-5}$ (3% warmup, linear decay), max length 2048 (7B) / 1524 (13B), DeepSpeed ZeRO-3, bf16, FlashAttention.

**Connection to the "Why does it work?" question.** Because the critic's judgments were *distilled into* $\mathcal{M}$, at inference $\mathcal{M}$ is its own reward model: $p_{\mathcal{M}}(\texttt{ISSUP=Fully}\mid x,d,y_t)$ is a learned estimate of "is $y_t$ entailed by $d$?". The paper's name, **self**-reflection, comes from this.

---

## 11. Inference (paper §3.3, Algorithm 1, App. A.3)

### 11.1 Algorithm 1 in plain pseudo-code

```python
def self_rag_generate(x, M, R, K=5, B=2, delta=0.2, w={"rel":1.0, "sup":1.0, "use":0.5}):
    beams = [ (prefix=[], score=0.0) ]
    while not all_finished(beams):
        candidates = []
        for prefix, score in beams:
            p_yes = normalized_prob(M, x, prefix, group="Retrieve", token="Yes")   # §11.2
            if p_yes > delta:                                                    # adaptive retrieval
                D = R.top_k(query=(x, last_segment(prefix)), k=K)
                for d in D:                                                      # K branches, run in parallel
                    rel   = M.critique(x, d, prefix, "ISREL")                    # predicted BEFORE the segment
                    y_t   = M.generate_segment(x, d, prefix)                     # greedy within a segment
                    sup   = M.critique(x, d, y_t,   "ISSUP")
                    use   = M.critique(x, y_t,      "ISUSE")
                    f     = seg_prob(y_t) + w["rel"]*rel + w["sup"]*sup + w["use"]*use   # Eq. 3-4
                    candidates.append((prefix + [d, y_t], score + f))
            else:
                y_t = M.generate_segment(x, None, prefix)
                candidates.append((prefix + [y_t], score + seg_prob(y_t)))
        beams = top_B(candidates, B)                                             # segment-level beam search
    return best(beams)
```

(Scores are written as summed along the path; the paper describes the beam step informally as "top-$B$ segment continuations at each timestamp".)

### 11.2 Adaptive retrieval with a threshold (App. A.3)

$$
\boxed{\;\text{retrieve at step } t \iff \frac{p(\texttt{Retrieve=Yes})}{p(\texttt{Retrieve=Yes})+p(\texttt{Retrieve=No})} > \delta\;}
$$

This is the restricted-softmax identity from §1.3: the probability of "Yes" **given that the model is choosing between Yes and No**.

- $\delta\to 0$: retrieve almost always (like standard RAG). The paper uses $\delta=0$ for ALCE/ASQA, where citations are mandatory.
- $\delta\to 1$: almost never retrieve.
- **Default: $\delta=0.2$.** Retrieval is triggered even when "No" is more likely, as long as "Yes" has ≥20% of the Yes/No mass. This biases toward retrieving on factual tasks.

**Soft vs hard.** "Hard constraints" means retrieving only when `Yes` is the argmax token, i.e. effectively $\delta=0.5$ on the Yes/No pair. The ablation (§15) shows this retrieves too rarely on PopQA (45.5 → 28.3).

### 11.3 Converting critique tokens to numbers

For each critique group $G$, convert its token distribution at segment $t$ into a scalar $s^G_t$. Generic form (paper Eq. 4's definition):

$$
s^G_t = \frac{p_t(\hat r)}{\sum_{i=1}^{N^G}p_t(r_i)},\quad \hat r = \text{most desirable token of } G .
$$

The exact forms from App. A.3:

$$
s^{\text{REL}} = \frac{p(\texttt{Relevant})}{p(\texttt{Relevant})+p(\texttt{Irrelevant})}\;\in[0,1]
$$

$$
s^{\text{SUP}} = \frac{p(\texttt{Fully})}{S}+0.5\cdot\frac{p(\texttt{Partially})}{S},\qquad S=p(\texttt{Fully})+p(\texttt{Partially})+p(\texttt{No})\;\Rightarrow s^{\text{SUP}}\in[0,1]
$$

This is the **expected support level** under the renormalized distribution, with values Fully=1, Partially=0.5, No=0.

$$
s^{\text{USE}} = \sum_{i=1}^{5} u_i\,\frac{p(\texttt{ISUSE}=i)}{S},\quad u=(-1,-0.5,0,0.5,1),\quad S=\sum_{i=1}^5 p(\texttt{ISUSE}=i)\;\Rightarrow s^{\text{USE}}\in[-1,1]
$$

This is the **expected utility** on a symmetric scale centred at the neutral rating 3.

**Why normalize at all?** The raw $p(\texttt{Relevant})$ also competes with ~32,000 ordinary vocabulary tokens. Renormalizing within the group removes that nuisance mass and gives a calibrated-looking score in a fixed range, which lets you add scores from different groups.

### 11.4 The segment score (paper Eqs. 3–4)

$$
\boxed{\;f(y_t, d, \text{Critique}) = p(y_t\mid x, d, y_{<t}) + \underbrace{\sum_{G\in\{\text{REL},\text{SUP},\text{USE}\}} w^G\, s^G_t}_{\mathcal{S}(\text{Critique})}\;}
$$

- **First term:** how likely (fluent) the segment is under the LM. (In the released code this is, to our reading, the length-normalized sequence probability $\exp\big(\frac{1}{|y_t|}\sum\log p\big)$, so it lies in $[0,1]$ and is comparable with the $s^G$ terms.)
- **Second term:** a **weighted linear reward** from self-critique.
- **Defaults:** $w^{\text{REL}}=1.0,\; w^{\text{SUP}}=1.0,\; w^{\text{USE}}=0.5$.
- **Customization:** raise $w^{\text{SUP}}$ to favour well-cited, conservative answers; lower it for more fluent, complete answers (see Fig. 3b in §15).
- **Hard filtering (alternative):** discard any candidate whose ISSUP is `No support` (a hard constraint instead of a soft reward).

### 11.5 Worked numeric example: why the "less fluent" candidate can win

Query: *"How did US states get their names?"* At segment $t$ the model retrieves $K=2$ passages:

| | $d_1$: "...11 of 50 states named after persons..." | $d_2$: "Popular baby names by state: In Texas, Emma..." |
|---|---|---|
| Segment probability $p(y_t\mid\cdot)$ | 0.60 | **0.70** |
| $p(\text{Rel}),\,p(\text{Irr})$ | 0.9, 0.1 → $s^{\text{REL}}=$ **0.90** | 0.2, 0.8 → $s^{\text{REL}}=$ **0.20** |
| $p(\text{Fully}),p(\text{Part}),p(\text{No})$ | 0.7, 0.2, 0.1 → $s^{\text{SUP}} = 0.7+0.5(0.2)=$ **0.80** | 0.1, 0.3, 0.6 → $s^{\text{SUP}} = 0.1+0.15=$ **0.25** |
| $p(\text{USE}=1..5)$ | .01,.04,.15,.40,.40 → $s^{\text{USE}}= -.01-.02+0+.20+.40=$ **0.57** | .05,.10,.25,.35,.25 → $s^{\text{USE}}=-.05-.05+0+.175+.25=$ **0.325** |
| $f = p + 1(s^{\text{REL}}) + 1(s^{\text{SUP}}) + 0.5(s^{\text{USE}})$ | 0.60+0.90+0.80+0.285 = **2.585** | 0.70+0.20+0.25+0.1625 = **1.3125** |

Plain likelihood decoding would choose $d_2$'s continuation (0.70 > 0.60), which is a fluent sentence grounded in an off-topic passage. Self-RAG chooses $d_1$. **This is the whole point:** the critique terms move the selection from "most fluent" to "most fluent *and* grounded".

**Check yourself:** with $w^{\text{REL}}=w^{\text{SUP}}=w^{\text{USE}}=0$, which candidate wins? (Answer: $d_2$. The method reduces to likelihood ranking.)

### 11.6 Closed-set tasks (PubHealth, ARC)

For classification/multiple choice, instead of taking the argmax sequence, the scores of all candidates are **aggregated per answer option**, and the option with the highest total wins (App. B.2).

### 11.7 Inference cost (be honest about it)

For each retrieval step, Self-RAG runs $K$ generations (one per passage) for each of $B$ beams, so roughly $B\cdot K$ segment generations per step, plus a few extra tokens for critiques. With $K=5$, $B=2$ that is ~10× the decoding work of one plain RAG pass on that segment, and each branch also prefills a passage. Mitigations used: vLLM batching (branches are independent and run in parallel), greedy decoding within segments, and **skipping retrieval altogether** when the Retrieve token says No. The paper does not report wall-clock latency, which is a fair criticism.

---

## 12. Why does it work? (the "insight" slide)

1. **Decomposition.** "Generate a good grounded answer" is split into four easier, *separately supervised* sub-decisions (need? relevant? supported? useful?). Each has a clean label, and a 7B model learns them well (critic accuracy 73–94%).
2. **Generate-then-verify beats generate-only.** Checking entailment of a short sentence against one passage is easier than generating a correct sentence. Over-generating $K$ candidates and picking with a verifier turns this asymmetry into accuracy (the same principle as best-of-$N$ sampling and reranking).
3. **One passage per candidate prevents distraction.** Instead of concatenating $K$ passages (and inheriting the noise of all of them), each candidate sees *one* passage. Irrelevant ones are scored down by ISREL rather than mixed in.
4. **Selective retrieval keeps versatility.** Creative and reasoning prompts skip retrieval, so they behave like a normal instruction-tuned LM.
5. **A precise connection to RLHF.** The optimum of KL-regularized RLHF, $\max_\pi \mathbb{E}_\pi[r]-\beta\,\mathrm{KL}(\pi\Vert\pi_{\text{ref}})$, has a closed form:
   $$\pi^*(y\mid x)=\frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\big(r(x,y)/\beta\big)\;\;\Longrightarrow\;\;\log\pi^*=\log\pi_{\text{ref}}+\tfrac{1}{\beta}r-\log Z.$$
   Ranking candidates by "LM likelihood term + $\sum_G w^G s^G$" is the same *shape* as ranking by $\log\pi^*$ with reward $r=\sum_G w^G s^G$. The difference is that Self-RAG uses $p$ rather than $\log p$, and searches over a finite candidate set. So **Self-RAG approximates decoding from a reward-tilted policy at test time**, and changing $w^G$ is like choosing a different reward *without retraining*. This is the mathematical reason the paper can claim "controllability" where RLHF needs a new training run.
6. **Contrast with Lewis-RAG marginalization.** Lewis-RAG computes $\sum_z p(z\mid x)p(y\mid x,z)$, a *soft average* over passages weighted by the *retriever*. Self-RAG instead does a *hard selection* weighted by the *generator's own judgment* of relevance and support. The retriever's score is only a first-stage filter.

---

# PART III: EXPERIMENTS

## 13. Setup: tasks, metrics, baselines

All evaluations are **zero-shot**: an instruction is given, with no in-context examples.

### 13.1 Six tasks in three families

| Family | Dataset | What it tests | Metric |
|---|---|---|---|
| Closed-set | **PubHealth** | public-health claim verification (true/false) | accuracy |
| Closed-set | **ARC-Challenge** | grade-school science multiple choice (reasoning) | accuracy |
| Short-form QA | **PopQA** (long-tail, 1,399 q) | rare-entity factual questions | "match" accuracy |
| Short-form QA | **TriviaQA-unfiltered** (11,313 q) | trivia factual questions | "match" accuracy |
| Long-form | **Biography generation** | "Tell me a bio about [person]" | FactScore |
| Long-form | **ASQA** (ALCE) | ambiguous questions needing multi-answer, cited, long responses | str-em, ROUGE, MAUVE, citation precision/recall |

### 13.2 Metric definitions (know these: the instructor may ask "what does this number mean?")

**Match accuracy (PopQA, TQA).** Exact match is too strict for free-form generation, so:
$$\text{Acc}=\frac{1}{N}\sum_{j=1}^{N}\mathbb{1}\big[\exists\,a\in\mathcal{A}_j:\ a \text{ is a substring of } \hat y_j\big],$$
where $\mathcal{A}_j$ is the set of gold answer aliases. *Caveat:* a long, rambling answer that mentions many candidates can be "correct" by accident.

**FactScore (Bio; Min et al. 2023).** Split the generation into **atomic facts** $\mathcal{A}_y$ (using an LLM), check each against Wikipedia (retrieval + LM verifier), and average:
$$\text{FS}(y)=\frac{1}{|\mathcal{A}_y|}\sum_{a\in\mathcal{A}_y}\mathbb{1}[a\ \text{supported by Wikipedia}],\qquad \text{FS}=\mathbb{E}_{x}\,[\text{FS}(y_x)].$$
It measures **factual precision**, not completeness. *Caveat:* a short bio with three safe facts can outscore a long, informative one. The paper itself notes that 7B sometimes beats 13B *because it writes shorter, precisely grounded outputs*.

**str-em (ASQA).** Each ambiguous question has several disambiguated short answers; str-em is the fraction of them that appear verbatim in the output (a measure of correctness/coverage).

**ROUGE (ASQA).** n-gram overlap with reference long answers (ROUGE-L uses the longest common subsequence).

**MAUVE (ASQA, fluency; Pillutla et al. 2021).** It compares the *distribution* of model text $Q$ with human text $P$ in an embedding space (quantized into clusters). Define mixtures $R_\lambda=\lambda P+(1-\lambda)Q$ and the divergence curve
$$\mathcal{C}(P,Q)=\Big\{\big(e^{-c\,\mathrm{KL}(Q\Vert R_\lambda)},\;e^{-c\,\mathrm{KL}(P\Vert R_\lambda)}\big):\lambda\in(0,1)\Big\}.$$
MAUVE is the **area under this curve**. It is 1 when $Q=P$ and near 0 when they are very different. Higher means more human-like.

**Citation recall & precision (ALCE; Gao et al. 2023).** For each output statement $s_i$ with cited passages $C_i$, an NLI model (TRUE, T5-11B) checks entailment:
- **Recall:** $\text{rec}_i=\mathbb{1}[\text{concat}(C_i)\models s_i]$. Is the statement *fully supported* by its citations?
- **Precision:** a citation $c\in C_i$ is *irrelevant* if $c$ alone does not support $s_i$ **and** removing it doesn't hurt ($C_i\setminus\{c\}\models s_i$). Precision is the fraction of citations that are *not* irrelevant (for supported statements).
- Recall answers "are claims backed?"; precision answers "is every citation pulling its weight?".

### 13.3 Baselines (and why each is the *right* comparison)

| Group | Models | What it controls for |
|---|---|---|
| No retrieval, public | Llama2-7B/13B, Alpaca-7B/13B (Llama2 replication) | the effect of retrieval + reflection |
| No retrieval, proprietary/RLHF | ChatGPT, Llama2-chat-13B | "is a 7B Self-RAG competitive with much bigger, RLHF-trained models?" |
| Prompt-engineering | CoVE-65B (Chain-of-Verification) | iterative self-verification by prompting vs learned reflection |
| Standard RAG | Llama2 / Alpaca / Llama2-chat / ChatGPT + same retrieved passages prepended | **the key comparison: same retriever, no reflection** |
| Same data, no reflection | **Llama2-FT-7B**: fine-tuned on Self-RAG's 150k pairs without reflection tokens or passages; retrieval added at test time | isolates "is the gain from data or from the framework?" |
| Trained with retrieval | SAIL-7B, Toolformer-6B (numbers copied from papers) | other retrieval-aware training schemes |
| Production | Perplexity.ai | a real search-LLM product (Bio only) |

### 13.4 Inference settings (defaults)

- Retriever: **Contriever-MS MARCO**, top-5 passages from Wikipedia (2018 dump; Dec 2020 dump for PopQA, because of newer entities).
- For Bio and open-domain QA: **5 extra documents from Google Programmable Search** (following SAIL).
- For ASQA: the ALCE authors' top-5 GTR-XXL documents, **for all systems** (fair comparison).
- $w^{\text{REL}}=1,\ w^{\text{SUP}}=1,\ w^{\text{USE}}=0.5$; $\delta=0.2$ ($\delta=0$ for ASQA); beam $B=2$; greedy decoding within each segment; vLLM.

---

## 14. Main results (paper Table 2): what to remember and how to read it

### 14.1 Condensed table

| Model | PopQA | TQA | PubHealth | ARC | Bio FS | ASQA em | ASQA mau | ASQA prec | ASQA rec |
|---|---|---|---|---|---|---|---|---|---|
| ChatGPT (no ret.) | 29.3 | **74.3** | 70.1 | **75.3** | 71.8 | 35.3 | 68.8 | – | – |
| Ret-ChatGPT | 50.8 | 65.7 | 54.7 | **75.3** | – | **40.7** | **79.7** | 65.1 | **76.6** |
| Llama2-chat-13B | 20.0 | 59.3 | 49.4 | 38.4 | 55.9 | 22.4 | 28.6 | – | – |
| Ret-Llama2-chat-13B | 51.8 | 59.8 | 52.1 | 37.9 | 79.9 | 32.8 | 43.8 | 19.8 | 36.1 |
| Alpaca-13B (no ret.) | 24.4 | 61.3 | 55.5 | 54.9 | 50.2 | 22.9 | 70.6 | – | – |
| Alpaca-13B + ret. | 46.1 | 66.9 | 51.1 | 57.6 | 77.7 | 34.8 | 56.6 | 2.0 | 3.8 |
| **Llama2-FT-7B** + ret. | 48.7 | 57.3 | 64.3 | 65.8 | 78.2 | 31.0 | 51.2 | 5.0 | 7.5 |
| SAIL-7B | – | – | 69.2 | 48.4 | – | – | – | – | – |
| CoVE-65B | – | – | – | – | 71.2 | – | – | – | – |
| **Self-RAG 7B** | 54.9 | 66.4 | 72.4 | 67.3 | **81.2** | 30.0 | 74.3 | 66.9 | 67.8 |
| **Self-RAG 13B** | **55.8** | 69.3 | **74.5** | 73.1 | 80.2 | 31.7 | 71.6 | **70.3** | 71.3 |

### 14.2 Five takeaways (use these on the results slide)

1. **Best open model on every task.** Self-RAG beats all non-proprietary baselines across all six tasks.
2. **Beats ChatGPT on some tasks, not all.** It wins on PopQA (55.8 vs 29.3/50.8), PubHealth (74.5 vs 70.1), Bio FactScore (81.2 vs 71.8), and citation precision (70.3 vs 65.1). It **loses** on TriviaQA (69.3 vs 74.3), ARC (73.1 vs 75.3), ASQA str-em (31.7 vs 40.7) and citation recall (71.3 vs 76.6). Say this honestly in the presentation; the paper's abstract wording is a bit generous.
3. **Standard RAG can *hurt*.** Retrieval helps baselines on PopQA and Bio, where you can copy answers from passages, but **hurts or barely helps** on PubHealth and ARC (ChatGPT: 70.1 → 54.7 on PubHealth; Alpaca-13B: 55.5 → 51.1). That is exactly the "indiscriminate retrieval" problem from §5. Self-RAG doesn't suffer from it.
4. **Citations: a jump of more than 10×.** Open RAG baselines have 2–8% citation precision/recall; Self-RAG has 67–71%, close to or above ChatGPT.
5. **It's the framework, not the data.** Llama2-FT-7B sees *the same 150k examples* and the same test-time retrieval, yet is far behind (e.g. PubHealth 64.3 vs 72.4, citation precision 5.0 vs 66.9). The reflection-token training is what makes the difference.

### 14.3 Grounding analysis (App. C.1)

Among *correct* answers on PopQA/TQA, how often does the answer string **not** appear in the retrieved passages (i.e. it came from parametric memory)? About **2% for Self-RAG** vs **15–20% for Alpaca / Llama2-chat**. Self-RAG's correct answers are much more often *traceable to evidence*.

---

## 15. Analysis & ablations (paper §5.2, Fig. 3 & 4)

### 15.1 Ablations (7B, trained on 50k examples for speed)

| Variant | PopQA | PubHealth | ASQA em | What it removes |
|---|---|---|---|---|
| **Self-RAG (50k)** | **45.5** | **73.5** | **32.1** | – |
| *Training:* No Retriever | 43.6 | 67.8 | 31.0 | plain instruction tuning, no passages in training |
| *Training:* No Critic | 42.6 | 72.0 | **18.1** | always top-1 passage in training, no reflection tokens (≈SAIL) |
| *Test:* No retrieval | **24.7** | 73.0 | – | never retrieve at inference |
| *Test:* Hard constraints | 28.3 | 72.6 | – | retrieve only if `Yes` is argmax (no threshold) |
| *Test:* Retrieve top-1 | 41.8 | 73.1 | 28.6 | always use only top-1 passage (standard RAG) |
| *Test:* Remove ISSUP | 44.1 | 73.2 | 30.6 | drop the support term from Eq. 4 |

**How to interpret (each line is a likely Q&A question):**
- **No Critic → ASQA em collapses 32.1 → 18.1.** Without reflection tokens, the model can't select among passages or produce grounded multi-sentence answers. *The critic is the most important training component.*
- **No retrieval → PopQA 45.5 → 24.7, but PubHealth barely moves (73.5 → 73.0).** Long-tail facts need retrieval, while PubHealth claims are more often answerable from parametric knowledge. This also explains why adaptive retrieval matters: needs differ by task.
- **Hard constraints → PopQA 28.3.** Using argmax instead of the soft threshold makes the model retrieve far less often (§11.2). The **threshold $\delta$ is a crucial knob**, not a detail.
- **Retrieve top-1 → PopQA −3.7, ASQA −3.5.** Trusting the retriever's top passage (standard RAG) is worse than letting the LM choose among $K$ by ISREL/ISSUP.
- **Remove ISSUP → ASQA −1.5.** Support scoring matters most for long-form, cited answers. The effect on PopQA/PubHealth is small (−1.4/−0.3).

### 15.2 Test-time customization (Fig. 3b, ASQA)

Increasing $w^{\text{SUP}}$ from 0 to 2: citation precision rises (~69.7 → **70.5**), while MAUVE fluency falls (~96 → **~86**). **Interpretation:** a classic precision/fluency trade-off. Longer, more fluent text contains more claims that aren't fully backed by citations. With Self-RAG you pick the point on this curve **at inference, with no retraining**.

### 15.3 Efficiency vs accuracy (Fig. 3c)

Sweeping $\delta$ changes retrieval frequency a lot on both PubHealth and PopQA. Retrieving less **hurts PopQA much more** than PubHealth, which is consistent with the "No retrieval" ablation. So practitioners can trade cost for accuracy per task.

### 15.4 Data scaling (Fig. 4a–c)

Training on 5k → 10k → 20k → 50k → 150k examples: PopQA rises from ~34 to ~55, and ASQA citation precision from ~41 to ~70. Both are still rising at 150k, which suggests **more data would help further**. PubHealth is roughly flat. Llama2-FT shows no comparable gain from 50k → 150k.

### 15.5 Human evaluation (Fig. 4d; only 50 samples per dataset)

| | PopQA | Bio |
|---|---|---|
| S&P (output Supported & Plausible) | 92.5 | 70.0 |
| ISREL prediction judged correct | 95.0 | 90.0 |
| ISSUP prediction judged correct | 90.0 | 85.0 |

Humans mostly agree with the model's self-assessment, so reflection tokens are *meaningful*, not decorative. Note that S&P excludes cases where Self-RAG itself predicted Irrelevant/No support.

---

# PART IV: CRITICAL ANALYSIS & IMPACT

## 16. Strengths, limitations, assumptions

### 16.1 Strengths

- **A simple mechanism with broad effect.** Vocabulary expansion + SFT + reranking; no RL, no extra inference-time model.
- **One model, many behaviours.** $\delta$ and $w^G$ give test-time control over retrieval frequency and the precision/fluency trade-off.
- **Interpretability and verifiability.** Every segment carries an explicit, mostly accurate (85–95% human agreement) label for relevance and support, plus a citation.
- **Strong controls.** The Llama2-FT baseline (same data) and the training/inference ablations isolate *where* the gains come from.
- **Breadth of evaluation.** Six tasks, three formats, multiple metrics including citations and fluency.
- **Open release.** Code, models and data are available, which supports reproducibility.

### 16.2 Limitations and weaknesses (pick 2–3 to argue in depth)

1. **Dependence on GPT-4 as the ultimate source of "truth" about reflection.** All critique labels trace back to GPT-4 judgments (temperature 1, a single sample). Any GPT-4 bias (e.g. towards verbose answers being "useful") is distilled twice. The human validation is tiny (20 items per type).
2. **Error cascade through two distillation steps.** The critic agrees with GPT-4 only 80% on ISREL and 73.5% on ISUSE. Those errors become *training targets* for $\mathcal{M}$, so the generator can confidently mislabel (Table 6: *"The Lie"* example, where the passage was about a different book but was labelled Relevant + Fully supported).
3. **The retriever is frozen and never improved.** If the right passage isn't in the top-$K$, reflection can only *detect* failure (ISREL=Irrelevant), not *fix* it. There is no query rewriting or re-retrieval loop. The authors themselves note that off-the-shelf retrievers are weak for open-ended instructions.
4. **The scoring function is a heuristic.** Eq. 3 adds a *probability* to *expected scores* on different ranges ([0,1] and [−1,1]). The usefulness scale $(-1,-0.5,0,0.5,1)$, the 0.5 credit for "partially supported", and the default weights $(1,1,0.5)$ are hand-picked. No principled calibration or sensitivity study is reported beyond $w^{\text{SUP}}$ and $\delta$.
5. **Inference cost is not reported.** Up to $B\cdot K=10$ segment generations per retrieval step, plus passage prefills. Latency/throughput vs standard RAG is never measured.
6. **The segment = sentence assumption.** Splitting with spaCy means a claim spanning two sentences, or a sentence needing two passages, is handled poorly. Each candidate segment is conditioned on **one** passage, so multi-hop synthesis across passages within a sentence is not directly supported.
7. **Evaluation caveats.**
   - Self-RAG uses Contriever **+ Google search** passages for Bio/open-domain QA. The text says retrieval settings are shared "across all baselines" explicitly only for ASQA, so check whether every retrieval baseline got the web results too.
   - Some baselines (SAIL, Toolformer, CoVE) have numbers copied from their papers, with different setups.
   - "Match" accuracy and FactScore both reward behaviours (long or short outputs) that aren't the same as quality. The paper itself notes 7B > 13B on factual precision because of shorter outputs.
   - Ablations use a 50k-trained model and 150 sampled ASQA instances; no variance or seeds are reported.
   - Human evaluation covers 50 examples and excludes cases the model itself flagged as irrelevant/unsupported.
8. **"Perceived utility" is decoupled from truth by design.** ISUSE can reward confident-sounding answers. Its weight is kept lower (0.5) but is never ablated.
9. **Self-assessment is not verification.** ISSUP is the model grading itself. If the model is confidently wrong, its self-label may be too (overconfidence/miscalibration was not studied).

### 16.3 Hidden assumptions

- A good answer can be built **sentence by sentence**, each sentence supported by a **single** passage.
- GPT-4-style judgments of relevance/support/utility are a valid proxy for human judgments.
- The corpus (Wikipedia 2018/2020 + web snippets) contains the needed facts.
- Token probabilities of critique tokens are meaningful (roughly calibrated) scores that can be added linearly.

### 16.4 How would you improve it? (have 2 ready)

- **Train the retriever jointly** or add **query reformulation / re-retrieval** when ISREL is low for all passages (later done by CRAG-style systems with a web-search fallback).
- **Learn the weights $w^G$** (e.g. on a small dev set per task) instead of hand-setting them, or calibrate the critique probabilities (temperature scaling).
- **Multi-passage segments:** let a segment condition on a small *set* of passages chosen by ISREL.
- **Replace SFT-on-labels with preference optimization** (e.g. DPO) using critique-based pairs, so the generator also *improves* its outputs, not just labels them.
- **Report latency** and use early pruning: stop expanding branches whose ISREL is low *before* generating the segment.
- **Use ensembles of LLM judges or human labels** for the critic data to reduce single-teacher bias.

## 17. Impact on later research and modern systems

- **Venue/recognition:** accepted at ICLR 2024 (oral). It is one of the most-cited RAG papers of 2023–24, and its framing of *"adaptive retrieval + self-critique"* became a standard reference point.
- **Direct follow-ups in the RAG literature:**
  - **CRAG (Corrective RAG, 2024):** a lightweight evaluator grades retrieved documents and triggers **web search / knowledge refinement** when they're poor. This addresses limitation #3 above.
  - **Adaptive-RAG (2024):** a classifier routes queries by complexity (no retrieval / single-step / multi-step), extending "retrieve only when needed".
  - **RankRAG-style work (2024):** one LLM trained to both rank contexts and generate. This is the same "single model does judgment + generation" philosophy.
- **Engineering practice:** LangChain/LangGraph and LlamaIndex ship "Self-RAG" and "Corrective RAG" recipes. In practice these are usually implemented *by prompting* an LLM to act as retrieval-grader / hallucination-grader / answer-grader. Those are exactly ISREL / ISSUP / ISUSE, as nodes in an agent graph.
- **Modern LLMs:** today's assistants treat search as a **tool the model chooses to call**. That is the Retrieve token, generalized. They are trained to cite sources, which is the ISSUP-style attribution. And reasoning models trained with RL to interleave thinking and searching (e.g. Search-R1-style work, 2025) push the "decide when to retrieve, inside the generation" idea further, replacing Self-RAG's offline labels with outcome-based RL.
- **Conceptual legacy:** *special tokens as an interface for control and self-evaluation*, and *test-time reward-weighted reranking instead of retraining*.

---

# PART V: EXAM PREP

## 18. Q&A bank (practice answering out loud in under 45 seconds each)

**Q1. In one sentence, what is Self-RAG?**
A single LM trained to emit special reflection tokens that decide when to retrieve and critique the relevance, support and usefulness of passages and its own output; those tokens are then used to guide decoding.

**Q2. Why not just always retrieve?**
It hurts versatility (creative/reasoning tasks) and adds distracting context. Table 2: Ret-ChatGPT drops 70.1 → 54.7 on PubHealth. The "No retrieval" ablation shows the need is task-dependent (PopQA −20.8, PubHealth −0.5).

**Q3. What does Eq. 1 mean?**
Maximum-likelihood training of the critic: maximize $\log p_\mathcal{C}(r\mid x,y)$, i.e. predict GPT-4's reflection label as text. It is a classifier implemented as conditional language modeling.

**Q4. What does Eq. 2 mean, and how is it different from Eq. 1?**
Standard next-token NLL over the *interleaved* sequence of answer text and reflection tokens, with passage tokens masked. The generator learns to produce both the answer and the critique; the critic produced only the critique.

**Q5. Why mask retrieved passages in the loss?**
They're inputs from the retriever, not model outputs. Predicting them would teach memorization/hallucination of passages. The model still attends to them.

**Q6. Why train a separate critic instead of using GPT-4 directly to label all data?**
Cost and reproducibility: the full annotation needs many calls per example (sentences × passages × 4 aspects). A 7B critic reaches 80–94% agreement on most aspects for a fraction of the cost.

**Q7. Why not use the critic at inference, or RLHF?**
The generator has internalized the critic, so one model suffices at inference. RLHF/PPO needs ~4 models in memory plus on-policy sampling, and fixes behaviour at training time. Self-RAG is plain SFT and is tunable at inference.

**Q8. Explain Eq. 3/4 with numbers.**
Segment score = segment probability + $1.0\,s^{\text{REL}}+1.0\,s^{\text{SUP}}+0.5\,s^{\text{USE}}$; each $s$ is the normalized desirable-token probability. Walk through §11.5, where the grounded candidate beats the more fluent one (2.585 vs 1.31).

**Q9. Why normalize critique probabilities within the group?**
To get $p(\hat r\mid r\in G)$: it removes probability mass on unrelated vocabulary tokens and puts each score in a fixed range, so the scores can be added.

**Q10. What is the threshold $\delta$, and why 0.2?**
Retrieve if $p(\text{Yes})/(p(\text{Yes})+p(\text{No}))>\delta$. A low $\delta$ biases toward retrieving on factual tasks. The "hard constraints" (argmax) ablation shows retrieving too little hurts PopQA badly (45.5 → 28.3).

**Q11. What happens if you remove the critic in training?**
ASQA em drops 32.1 → 18.1 and PopQA 45.5 → 42.6. That is the largest training ablation effect, so reflection tokens are the core.

**Q12. Is the gain just from better training data?**
No. Llama2-FT-7B is trained on the same 150k pairs and retrieves at test time, but is much worse (PubHealth 64.3 vs 72.4, citation precision 5.0 vs 66.9).

**Q13. Why is Llama2-FT an appropriate baseline?**
It is a controlled experiment: it changes only the presence of reflection tokens/passages in training, holding the data and base model fixed.

**Q14. Why does 7B sometimes beat 13B?**
On FactScore (81.2 vs 80.2), the 7B model tends to write shorter, precisely grounded outputs. FactScore measures precision, not coverage.

**Q15. Does Self-RAG beat ChatGPT?**
On PopQA, PubHealth, Bio, and citation precision, yes. On TriviaQA, ARC, ASQA str-em and citation recall, no. Remember that ChatGPT is far larger and RLHF-trained.

**Q16. What does the $w^{\text{SUP}}$ trade-off show?**
A higher support weight gives higher citation precision but lower MAUVE fluency. Behaviour is customizable at inference without retraining.

**Q17. Biggest limitation?**
(Pick one and defend it.) Reliance on GPT-4-derived labels through two distillation steps, with small human validation. *Or:* a frozen retriever with no recovery when all passages are irrelevant. *Or:* unreported inference cost from $B\cdot K$ branching.

**Q18. What assumption does segment-level generation make?**
That each sentence can be supported by a single passage, and that sentence boundaries are the right granularity for retrieval decisions.

**Q19. How is this different from FLARE (Active RAG)?**
FLARE decides retrieval from low token confidence of a prompted proprietary LLM, with no critique. Self-RAG *learns* explicit retrieve/relevance/support/utility tokens via training and uses them for reranking.

**Q20. How is this different from Lewis-RAG?**
Lewis-RAG always retrieves and marginalizes over passages using retriever scores (a soft mixture). Self-RAG retrieves adaptively, generates one candidate per passage, and *selects* using the LM's own relevance/support judgments.

**Q21. How does this relate to modern LLMs?**
Tool-calling search is the generalized Retrieve token; citation-grounded answers and grader nodes in agentic RAG (LangGraph Self-RAG/CRAG) mirror ISREL/ISSUP/ISUSE; RL-trained search-reasoning models replace offline labels with outcome rewards.

**Q22. Why is there a "Continue" option for Retrieve?**
A passage may support several consecutive sentences (e.g. a biography from one Wikipedia intro). "Continue" reuses it and avoids redundant retrieval calls.

**Q23. Why is ISUSE weighted only 0.5?**
It measures *perceived* helpfulness independent of factuality, and its critic accuracy is the lowest (73.5%). Down-weighting stops fluent-but-unsupported answers from winning.

**Q24. Is the experimental evidence sufficient for the claims?**
For "better than open RAG baselines": yes, convincingly (many tasks, the FT control, ablations). For "outperforms ChatGPT": only partly, since it is task-dependent. Missing: latency, seeds/variance, weight sensitivity beyond ISSUP, larger human evaluation.

**Q25. What does the training-data-scaling result imply?**
Performance is still rising at 150k on PopQA and ASQA, so the method is data-hungry for reflection-token learning and would likely improve with more annotated data.

**Q26. Why does Self-RAG rely less on parametric memory?**
Only ~2% of its correct answers are absent from the retrieved passages, vs 15–20% for baselines. It was trained to tie each sentence to evidence and flag irrelevant evidence.

## 19. Self-check exercises (solutions below; try them first)

**E1.** A Retrieve-token distribution at some step gives $p(\text{Yes})=0.12$, $p(\text{No})=0.40$, $p(\text{Continue})=0.03$, and the remaining mass goes to other vocabulary tokens. With $\delta=0.2$, do we retrieve? With "hard constraints"?

**E2.** ISSUP probabilities: Fully 0.30, Partially 0.50, No 0.10 (the rest is elsewhere). Compute $s^{\text{SUP}}$.

**E3.** ISUSE probabilities for 1..5: (0.0, 0.1, 0.2, 0.3, 0.4). Compute $s^{\text{USE}}$. What are its minimum and maximum possible values?

**E4.** In §11.5, find the smallest $w^{\text{REL}}$ (holding the other weights at defaults) for which $d_2$ would win. Is it possible?

**E5.** Write the masked loss for the sequence `[Retrieve=Yes] <p> Paris is the capital. </p> [Relevant] Paris [Fully] [Utility:5]`. Which tokens have $m_i=1$?

**E6.** A bio has 10 atomic facts; 8 are supported by Wikipedia. A second bio has 4 facts, all supported. FactScores? What does this tell you about the metric?

**E7.** Show that the group-normalized probability $p(r\mid r\in G)$ equals a softmax over only the logits of $G$.

**E8.** Why can't Eq. 2 be used to train the retriever $\mathcal{R}$?

### Solutions

**E1.** $0.12/(0.12+0.40)=0.231>0.2$, so **retrieve**. With hard constraints, the argmax is No (0.40 > 0.12), so **don't retrieve**. This is why hard constraints retrieved less in the ablation.

**E2.** $S=0.9$; $s^{\text{SUP}}=0.30/0.9+0.5\cdot0.50/0.9=0.333+0.278=0.611$.

**E3.** $S=1$; $s=-1(0)-0.5(0.1)+0(0.2)+0.5(0.3)+1(0.4)=-0.05+0.15+0.4=0.50$. The range is $[-1,1]$: −1 when all mass is on "1", +1 when all mass is on "5".

**E4.** $f(d_1)-f(d_2)=(0.60-0.70)+w^{\text{REL}}(0.90-0.20)+(0.80-0.25)+0.5(0.57-0.325)=-0.10+0.70w^{\text{REL}}+0.55+0.1225$. Setting this $<0$ needs $0.70w^{\text{REL}}<-0.5725$, i.e. $w^{\text{REL}}<-0.818$. Only a *negative* relevance weight (rewarding irrelevance) would flip it, so with any non-negative weights $d_1$ wins: ISSUP alone already outweighs the fluency gap.

**E5.** $m_i=1$ for `[Retrieve=Yes]`, `[Relevant]`, `Paris`, `[Fully]`, `[Utility:5]` (and `<p>`/`</p>` markers depending on the implementation). $m_i=0$ for the passage tokens "Paris is the capital.". The prompt $x$ is also excluded from the loss.

**E6.** 0.8 and 1.0. The shorter bio scores higher even though it's less informative: FactScore measures precision, not recall. This is why 7B > 13B on Bio is not a clear win.

**E7.** $p(r)=e^{z_r}/Z$ with $Z=\sum_{u\in V\cup R}e^{z_u}$, so $\frac{p(r)}{\sum_{r'\in G}p(r')}=\frac{e^{z_r}/Z}{\sum_{r'\in G}e^{z_{r'}}/Z}=\frac{e^{z_r}}{\sum_{r'\in G}e^{z_{r'}}}$. $Z$ cancels.

**E8.** Passages enter as discrete, non-differentiable top-$K$ selections inserted into the text, and their tokens are masked. No gradient flows to $\mathcal{R}$'s parameters (unlike Lewis-RAG, where $p_\eta(z\mid x)$ appears inside the marginal likelihood).

## 20. Suggested 15-minute slide plan mapped to the rubric

| # | Slide | Rubric | Time | Content (from this guide) |
|---|---|---|---|---|
| 1 | Title + one-line idea | – | 0:20 | §8 one-sentence summary |
| 2 | LMs hallucinate; RAG basics | Problem (2) | 1:30 | §2, §4.2 formula, PopQA long-tail numbers |
| 3 | Why vanilla RAG fails | Problem | 1:30 | §5's three failures, with Table 2 evidence (PubHealth drop, 2–8% citations) |
| 4 | Core idea: reflection tokens | Core (3) | 1:30 | §8.1 table + §8.2 example |
| 5 | What's novel / why it works | Core | 1:30 | §8.3, §12 (generate-then-verify, RLHF connection) |
| 6 | Formal model + token order | Technical (4) | 1:30 | §9.2 factorization, §9.3 |
| 7 | Training: critic → generator | Technical | 2:00 | §10 diagram, Eq. 1, Eq. 2 with mask, why not PPO |
| 8 | Inference: threshold + scoring | Technical | 2:00 | §11.2–11.4 + §11.5 numeric example |
| 9 | Setup + main results | Experiments (2) | 1:30 | §13 table, §14.2 five takeaways |
| 10 | Ablations + customization | Experiments | 1:00 | §15.1 interpretation, Fig. 3b trade-off |
| 11 | Strengths / limitations / improvements | Critical (2) | 1:00 | §16 (2 strengths, 3 limitations, 2 fixes) |
| 12 | Impact | Impact | 0:30 | §17 |
| 13 | **Team Contribution** (required) | – | 0:10 | table per `eval_rubric.md` §2 |

**Rubric reminder:** "If you put something on a slide, you should be able to explain why it is there." Every equation on slides 6–8 should come with a *why*. §9.3, §10.4 and §11.3 contain those whys.

## 21. Glossary

- **Attribution / citation:** linking a generated claim to the source passage that supports it.
- **Beam search:** keep the top-$B$ partial hypotheses at each step.
- **Bi-encoder / dense retriever:** separate encoders for queries and passages; relevance = dot product.
- **Contriever:** a dense retriever pre-trained with unsupervised contrastive learning; the "MS MARCO" variant is fine-tuned on search data.
- **Critic ($\mathcal{C}$):** a Llama2-7B model trained to predict reflection tokens; used only to label training data.
- **Distillation:** training a student model on a teacher's outputs.
- **Entailment / NLI:** whether text A logically supports text B.
- **FactScore:** fraction of atomic facts in a generation supported by a knowledge source.
- **Generator ($\mathcal{M}$):** the final Self-RAG model that writes answers *and* reflection tokens.
- **Hallucination:** fluent but unsupported/false output.
- **Instruction tuning / SFT:** supervised fine-tuning on instruction–response pairs.
- **MAUVE:** a distribution-level similarity between model and human text (fluency/naturalness).
- **MIPS:** maximum inner product search, the operation behind dense retrieval.
- **Parametric vs non-parametric memory:** knowledge in weights vs knowledge in an external corpus.
- **PPO / RLHF:** reinforcement learning from human feedback with proximal policy optimization.
- **RAG:** retrieval-augmented generation: condition the LM on retrieved passages.
- **Reflection tokens:** the new vocabulary items Retrieve, ISREL, ISSUP and ISUSE.
- **Segment:** the unit of generation and critique; one sentence in this paper.
- **Teacher forcing:** training on the gold prefix rather than the model's own outputs.

---

*Primary source for every number above: the paper (`SelfRAG_2310.11511.pdf`; see `paper_markdown/` for the text version). Statements about later work (§17) are context, not claims from the paper; verify specifics before putting them on a slide.*
