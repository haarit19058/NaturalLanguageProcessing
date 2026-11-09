# Self-RAG (detailed deck v3): speaker script

Read in a natural voice; times are approximate. Slides 31-32 are backup for Q&A and are not presented.
Estimated speaking time for slides 1-30: about 23 min 0 s.


## Slide 1: Self-RAG  (~30 s)

Good morning. We are presenting Self-RAG: Learning to Retrieve, Generate and Critique through Self-Reflection, by Akari Asai and colleagues from the University of Washington, the Allen Institute for AI and IBM, published at ICLR 2024.

In one sentence: Self-RAG is a single language model that decides for itself when to look something up, judges whether what it found is relevant, and checks whether its own sentences are supported. It does all of this by writing special tokens, so judging costs no more than writing a word.

## Slide 2: LLMs make things up  (~45 s)

Let's start with the problem. A language model stores its knowledge in its weights. That knowledge is frozen after training, comes with no source, and is hard to update.

Rare facts are the worst case. The model saw few training examples, so its probability is spread over answers that sound plausible but are wrong.

The numbers on the right come from the paper's Table 2. On PopQA's long-tail questions, Llama2-13B scores 14.7 percent without retrieval and 45.7 percent with it. So retrieval clearly matters, and the standard fix is retrieval-augmented generation, RAG. But as we'll see, RAG brings problems of its own.

## Slide 3: RAG in one slide  (~45 s)

Here is RAG in one picture. The retriever embeds the query and every passage separately and ranks passages by dot product. Self-RAG uses Contriever-MS MARCO and keeps it frozen. The top-K passages then go to the language model.

There are two classic ways to use them. Lewis et al. treat the passage as a latent variable and average the answer over passages, weighted by the retriever. In practice, most systems just paste all K passages into the prompt; that is the 'Ret-ChatGPT' baseline you will see later.

Both versions retrieve every single time, and neither checks the answer against the evidence.

## Slide 4: Why plain RAG falls short  (~50 s)

That gives three failure modes. One: RAG always retrieves, even for 'write an essay about your summer'. Two: the top passage can be off-topic, and models are easily distracted by irrelevant context. Three: nothing forces the answer to follow the passages, and nothing checks it afterwards.

The paper's own numbers show the damage. ChatGPT with retrieval drops from 70.1 to 54.7 on PubHealth. Alpaca-13B drops from 55.5 to 51.1. And open RAG models get only 2 to 6 percent citation precision on ASQA.

Earlier work fixes one piece each: FLARE retrieves when confidence is low, SAIL always prepends passages, Toolformer learns API calls, and RLHF needs expensive PPO training.

## Slide 5: RAG vs Self-RAG: the paper's picture  (~60 s)

This is the paper's Figure 1, and it summarises the whole method.

On the left, plain RAG retrieves three passages for 'How did US states get their names?', pastes them all in, and the answer mixes them up: one claim is contradicted by the passages and another has no support at all.

On the right, Self-RAG. Step 1: it writes a first sentence, then decides it needs evidence and emits Retrieve. Step 2: it writes one continuation per passage, in parallel, and labels each passage Relevant or Irrelevant. Step 3: it critiques each continuation, Supported or Partially supported, and keeps the best one. Here passage 1 beats passage 3, which beats passage 2. Then it repeats for the next sentence.

The bottom row: for an essay prompt, Self-RAG writes 'No Retrieval' and simply answers, while plain RAG still retrieves.

## Slide 6: Core idea: reflection tokens  (~60 s)

This is the central idea: reflection tokens. We add special tokens to the vocabulary and train one model, with ordinary next-token prediction, to write them between its own sentences. Critique becomes next-token prediction.

There are four types, from the paper's Table 1. Retrieve: should I fetch evidence for the next sentence? It can say Yes, No, or Continue, meaning reuse the passage I already have. IsRel: is this passage relevant to the question? IsSup: is every claim in my sentence backed by the passage, fully, partially, or not at all? IsUse: is the whole answer useful, on a 1-to-5 scale, regardless of whether it is true.

The first three are written for every sentence; IsUse once at the end. The bottom row of each card shows what that token is allowed to see.

## Slide 7: What the model actually writes  (~45 s)

Here is what the model actually writes, taken from the paper's training examples. For the Walking Dead question it writes Retrieve=Yes; the retriever inserts a passage; then the model writes IsRel, the answer, IsSup and IsUse. For the poem it never retrieves. Nobody hard-coded that; it is learned from the data.

The order matters because the model is autoregressive: earlier tokens condition later ones. Retrieve comes before writing. IsRel comes before the sentence, so the judgment shapes the answer. IsSup comes after, because you can only check a finished sentence. IsUse is last, because it is about the whole answer.

And remember: the passage itself comes from the retriever, not from the model.

## Slide 8: Reflection tokens, formally  (~60 s)

Now the same thing formally. For each segment t, the model first samples the Retrieve token, given the question and everything written so far. Only if it says Yes do we call the retriever; that is an external step, not part of the model. Then the model writes IsRel, the sentence, and IsSup, each conditioned on everything before it. After the last segment it writes IsUse.

Because it is all one token stream, the probability of the output factorises by the chain rule, shown at the bottom. The indicator switches the critique factors on only for segments where we retrieved. Passages are not in the product, because the model does not produce them.

On the right is the trick used everywhere at inference. To read a decision, we renormalise the probability inside the token group. The softmax denominator over 32,000 normal tokens cancels, so we get a clean probability for Yes versus No. The new tokens are just extra rows in the embedding and output layers.

## Slide 9: Training: distil a critic, then teach the generator  (~45 s)

How do we get training data with these tokens? Nobody has it, so the authors build it in two stages.

Stage one: GPT-4 labels a sample using few-shot prompts, one prompt per token type, and a Llama2-7B critic is fine-tuned to imitate those labels.

Stage two: the critic and the retriever annotate 150 thousand ordinary instruction and QA examples, inserting passages and reflection tokens. The generator is then trained on that with plain next-token prediction.

GPT-4 and the critic are only used to build the data. At inference, only the generator and the retriever run. The generator has learned to be its own critic.

## Slide 10: Stage 1: GPT-4 teaches the critic  (~50 s)

Stage one in numbers. On the left, the GPT-4 labels collected: about 12.6 thousand for Retrieve, 19.3 thousand for IsRel, 11.2 thousand for IsSup and 3.8 thousand for IsUse; about 47 thousand in total.

Is GPT-4 right? The authors checked 20 labels per type by hand: 95 percent agreement for Retrieve and IsRel, 90 for IsSup, 80 for IsUse.

On the right, how well the critic imitates GPT-4 on held-out labels. The Llama2-7B critic reaches about 94 percent on Retrieve and IsSup, 80 on IsRel and 73.5 on IsUse. A smaller FLAN-3B critic is clearly worse on IsSup, 73 versus 93.5. IsUse is the hardest for both; most errors are 4-versus-5 confusions, where humans disagree too.

## Slide 11: Stage 2: the critic labels 145k examples  (~60 s)

Stage two, from the paper's Figure 2 and Algorithm 3. On the left, a creative prompt: the critic says No Retrieval, and only a utility token is added. On the right, a factual answer: the critic says Retrieve, the retriever finds passages, and the critic labels them Relevant and Supported, or Partially.

The five steps below are the full procedure. Two details matter. First, step 4: if no passage is both relevant and supported, a random passage is inserted anyway. So the generator also sees Irrelevant and No-support examples, and learns to say so honestly. Second, step 5: the answer text is never rewritten. The critic only inserts tokens and passages around the original answer. The generator learns how to annotate good answers, not new answer content.

## Slide 12: The two training objectives  (~45 s)

These are the two objectives. Equation 1, the critic: maximise the log-probability of GPT-4's label given the input and output. It is a classifier written as language modelling.

Equation 2, the generator: maximise the log-probability of the answer and the reflection tokens together, interleaved, given the input.

Why not RLHF? PPO needs about four models in memory and fresh samples during training. Here we have one model, a fixed dataset, and cross-entropy. And behaviour can still be changed later, at inference, without retraining.

## Slide 13: The generator loss, token by token  (~45 s)

Equation 2 at the token level. It is the normal next-token loss with a mask: m equals zero for every token inside the retrieved passage and one otherwise.

The strip in the middle is one real training sequence. Green tokens get loss; the grey passage is masked.

Why mask the passage? The retriever supplies it at test time, so the model never needs to write it. Training on it would teach the model to memorise Wikipedia, or to invent passage-like text, which is exactly hallucination. The model still reads the passage through attention, so it learns to use it without learning to produce it.

The only architectural change is the bigger vocabulary. The training recipe on the right is from Appendix A.2.

## Slide 14: Inference, one sentence at a time  (~45 s)

Inference goes one sentence at a time. At each sentence boundary we read the Retrieve token as the probability of Yes within Yes and No, and retrieve if that is above delta.

The default delta is 0.2, so the model retrieves even when No is more likely. For ASQA it is 0, because citations are required.

If it retrieves, it fetches five passages, writes one candidate sentence per passage in parallel, critiques each candidate, and keeps the best two partial answers. That is a beam search over sentences, with greedy decoding inside each sentence.

## Slide 15: Algorithm 1, line by line  (~45 s)

This is the paper's Algorithm 1, so you can map each line. Lines 2 and 3: predict Retrieve; in practice the threshold from the previous slide, not the argmax. Line 4: retrieve passages using the question and the previous segment. Line 5: for each passage, predict IsRel, then the segment. Lines 6 and 7: predict IsSup and IsUse, and rank the candidates; the ranking formula is on the next slide. Lines 8 to 10: with no retrieval, it is just a normal language model.

The cost is real: up to two beams times five passages, so ten drafts per retrieval step. They run in parallel with vLLM, but the paper reports no latency numbers. For PubHealth and ARC, the scores are summed per answer option and the best option wins.

## Slide 16: Scoring a candidate sentence  (~60 s)

This is how a candidate sentence is scored: Equations 3 and 4. The first term is the sentence's own probability, how fluent it is. Then we add a weighted sum of critique scores.

Each score is an expected value under the renormalised token distribution. s-Rel is the probability of Relevant. s-Sup gives Fully 1, Partially a half and No 0. s-Use maps the 1-to-5 rating onto minus 1 to plus 1. Default weights are 1, 1 and 0.5.

The worked example uses the US-states prompt from Figure 1, with illustrative probabilities. Passage d2, about baby names, gives a slightly more fluent sentence: 0.70 versus 0.60. But after the critique terms, d1 scores 2.59 against 1.31. Likelihood alone would pick the off-topic sentence; the critique terms pick the grounded one. With all weights at zero, we are back to plain likelihood ranking.

## Slide 17: Why it works  (~45 s)

Why does this work? First, decomposition: four small, separately supervised decisions instead of one hard one. Second, verifying is easier than generating: checking a sentence against one passage is easier than writing a correct one. Third, one passage per candidate, so noisy passages are never mixed and bad ones are scored down. Fourth, retrieval only when useful, so essays and maths stay normal. Fifth, the critique is cheap: learned offline with supervised learning, with no reward model at test time.

Sixth, mathematically it resembles RLHF. The optimal RLHF policy is the reference policy reweighted by exp of the reward. Self-RAG reranks candidates by likelihood plus a critique reward, so changing the weights is like choosing a different reward, without retraining.

## Slide 18: Experimental setup  (~40 s)

Now the experiments. Six tasks in three families, all zero-shot. Closed-set: PubHealth and ARC-Challenge. Short-form QA: PopQA's rare entities and TriviaQA. Long-form: biographies, and ASQA, which needs citations.

The baselines come in four groups. The two that matter most are standard RAG, which is the same models with the same passages, and Llama2-FT, which is trained on the same 150 thousand examples but without reflection tokens. Llama2-FT tells us whether the gains come from the data or from the method.

## Slide 19: What each metric measures  (~50 s)

Before the results, here is what each number means. PopQA and TriviaQA count an answer correct if any gold alias appears in the output; a long answer can get lucky.

FactScore splits a biography into atomic facts and checks each against Wikipedia. It measures precision, so short bios do well.

ASQA citations are judged by an NLI model. Recall asks whether a statement is entailed by its citations. Precision asks whether each citation is actually needed: a citation is irrelevant if it does not support the statement alone and removing it changes nothing.

MAUVE compares the distribution of model text with human text; 1 means identical. str-em is the share of short gold answers that appear in the output.

## Slide 20: Main results  (~50 s)

The headline results. Self-RAG 13B is the best open model on all six tasks, as the paper claims. One metric-level exception, ASQA exact match, comes up on the next slide.

Against ChatGPT the picture is mixed. Self-RAG wins on PopQA, PubHealth, biographies and citation precision, but loses on TriviaQA, ARC and ASQA correctness. We want to be precise here, because the abstract is a little generous.

The key control is Llama2-FT: same data, same retrieval at test time, and it is far behind: 64.3 versus 72.4 on PubHealth, and 5 versus 67 on citation precision. So the gain comes from the framework, not the data.

## Slide 21: Table 2 in full  (~40 s)

This is the full Table 2 for reference; we won't read it out. Yellow marks the best non-proprietary score in each column, and the Self-RAG rows are green.

Three things to take from it. One: Self-RAG has the best open score in every column except ASQA exact match, where Alpaca-13B with retrieval is higher, 34.8 versus 31.7. Two: retrieval helps the baselines a lot on PopQA and Bio, but not on PubHealth. Three: look at the last two columns. Every open RAG baseline has citation precision and recall below 8 percent, except Llama2-chat at 20 and 36. Self-RAG is at 67 to 71, in ChatGPT's range.

## Slide 22: Retrieval helps some tasks, hurts others  (~45 s)

This chart takes Table 2 and computes how many points retrieval adds for each baseline. On PopQA every model gains about 21 to 32 points, because the answer can be copied straight out of a passage.

On PubHealth the gains vanish or turn negative: ChatGPT loses 15.4 points and Alpaca-7B loses 9.6. ARC is mixed and mostly small. These tasks need reasoning, and irrelevant passages distract. ChatGPT even loses 8.6 points on TriviaQA.

This is the paper's argument for deciding retrieval per query. Self-RAG's own ablation agrees: switching retrieval off costs 21 points on PopQA but only half a point on PubHealth.

## Slide 23: Citations and grounding  (~45 s)

Two views of grounding. On the left, ASQA citation precision and recall. Every open RAG baseline sits between 2 and 8 percent. Llama2-chat with retrieval reaches 20 and 36. Self-RAG reaches 67 to 71: comparable to ChatGPT with retrieval, and above it on precision, though below it on recall.

On the right, from Appendix C.1: among correct answers on PopQA and TriviaQA, how many are not found in the retrieved passages, so they came from memory? About 2 percent for Self-RAG, versus 15 to 20 percent for Alpaca and Llama2-chat. Self-RAG's correct answers are traceable to evidence.

One more detail: on biographies the 7B model beats 13B, because it writes shorter but fully grounded bios.

## Slide 24: Ablations: every piece matters  (~60 s)

Ablations, using a 7B model trained on 50 thousand examples for speed.

Training side: removing the critic, meaning always the top passage and no reflection tokens, similar to SAIL, collapses ASQA from 32.1 to 18.1. That is the single biggest drop, so the reflection tokens are the core of the method.

Test side: never retrieving costs 21 points on PopQA and almost nothing on PubHealth. Hard constraints, retrieving only when Yes is the argmax, retrieves too rarely and drops PopQA to 28.3, so the soft threshold matters. Always using the top-1 passage, which is standard RAG, loses about 3.5 points. Dropping IsSup mainly hurts long-form ASQA.

## Slide 25: Test-time knobs (Figure 3b, 3c)  (~45 s)

The paper's Figure 3b and 3c show the test-time knobs.

In 3b, raising the weight on IsSup from 0 to 2 raises citation precision from about 69.8 to 70.5 (these values are read off the plot), but MAUVE fluency falls from about 96 to 86. Longer, more fluent text contains more claims that are not fully backed. It is a precision-versus-fluency trade-off, and you choose the point at inference time.

In 3c, raising delta lowers how often the model retrieves. On PubHealth, accuracy barely moves even when retrieval drops to zero. On PopQA, accuracy falls together with retrieval. So you can trade cost for accuracy per task, without retraining.

## Slide 26: More data, and a human check (Figure 4)  (~45 s)

Figure 4. Panels a to c train the 7B model on 5 to 150 thousand examples. PopQA rises from about 34 to 55 and ASQA citation precision from about 41 to 70, both still rising, so more data would likely help. PubHealth stays roughly flat. The paper also notes that Llama2-FT does not improve comparably from 50k to 150k, which again points to the framework.

Panel d is a human check on 50 samples per dataset. 92.5 percent of PopQA outputs and 70 percent of biographies are supported and plausible (the paper excludes outputs that Self-RAG itself flagged as irrelevant or unsupported), and humans agree with the model's own IsRel and IsSup labels 85 to 95 percent of the time. The caveat: 50 samples is small.

## Slide 27: Critical analysis  (~60 s)

Our critical view. Strengths: no reinforcement learning, just supervised training plus reranking; behaviour is tunable at test time; every sentence comes with a citation and a self-assessed support label; and the controls, Llama2-FT and the ablations, are strong.

Limitations: every label traces back to GPT-4, checked on only 20 examples per type. The critic's errors, 80 percent accuracy on IsRel and 73.5 on IsUse, become training targets. The retriever is frozen, so the model can detect a bad passage but cannot fix it by searching again. And cost: up to ten drafts per step, with no latency reported.

Assumptions: one sentence comes from one passage, and probabilities from different token groups can simply be added. Improvements we would suggest: re-query when every IsRel is low, learn the weights on a dev set, let a sentence use several passages, or use preference training on critique pairs.

## Slide 28: Impact  (~40 s)

Impact. This slide is context from after the paper, not from the paper itself. Self-RAG sits between the 2020 RAG paper, which always retrieves, and today's systems. Follow-ups such as Corrective RAG grade the retrieved documents and fall back to web search, which fixes the frozen-retriever limitation; Adaptive-RAG routes queries by difficulty.

LangGraph and LlamaIndex ship Self-RAG recipes in which an LLM plays retrieval, hallucination and answer 'graders': the same roles as IsRel, IsSup and IsUse. Modern assistants decide when to search and cite.

The lasting idea: special tokens as an interface for control and self-evaluation, and reranking at test time instead of retraining.

## Slide 29: Team Contribution  (~15 s)

This is our team contribution, as the rubric requires. [Each member says one sentence about what they did.]

## Slide 30: Thank you!  (~10 s)

Thank you. We are happy to take questions.

[Prep, not to read aloud. Likely questions: Why mask the passages? Why a separate critic instead of GPT-4 directly? What do Equations 3 and 4 mean? Why delta = 0.2? What happens without the critic (ASQA 32.1 to 18.1)? Is the gain from data or framework (Llama2-FT)? Biggest limitation? How does this relate to modern LLMs? The two backup slides cover data and settings.]

## Slide 31: Backup: the training data (Table 3)  (backup)

Backup for Q&A. This is Table 3, the generator's training data: 145,619 examples, which the paper rounds to 150 thousand. About 92 thousand are instruction-following, from Open-Instruct: GPT-4 Alpaca, Stanford Alpaca, FLAN-V2, ShareGPT and Open Assistant. About 54 thousand are knowledge-intensive: Wizard of Wikipedia, Natural Questions, FEVER, OpenBookQA, ARC-Easy and ASQA. The knowledge-intensive part is what teaches retrieval and citation.

A fair point if asked: ARC-Easy and ASQA training data are related to the ARC-Challenge and ASQA test sets, so those two tasks are not fully out-of-domain.

## Slide 32: Backup: settings to know for Q&A  (backup)

Backup for Q&A: the exact settings, from Section 4.2 and Appendices A.2 and B.2. Use this slide if someone asks how many passages, which thresholds or weights, or what hardware was used.
