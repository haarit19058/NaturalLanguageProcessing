# Self-RAG presentation: speaker notes

## Slide 1: Self-RAG

Introduce the paper: Self-RAG from UW, AI2 and IBM, published at ICLR 2024. One-line pitch: a single language model that decides for itself when to look things up, judges what it found, and grades its own sentences, using special tokens in its vocabulary.

## Slide 2: LLMs make things up

Hallucination: models are trained to be plausible, not true. Parametric memory = weights; non-parametric = an external document collection. PopQA long tail (Mallen et al. 2023, same group): Llama2-13B gets 14.7% without retrieval and 45.7% with it, so retrieval matters most for rare facts. RAG is the fix everyone uses, and this paper is about its weaknesses.

## Slide 3: RAG in one slide

Walk the pipeline left to right. Dense retrieval: two encoders, dot-product similarity, nearest-neighbour search. Lewis-RAG marginalises over passages (soft mixture weighted by the retriever). In practice people just concatenate the top-K passages into the prompt. Both always retrieve, and neither checks that the answer follows the evidence.

## Slide 4: Why plain RAG falls short

Three failure modes from the introduction. Evidence from Table 2: retrieval drops ChatGPT on PubHealth from 70.1 to 54.7; Alpaca-13B from 55.5 to 51.1. Citation precision of open RAG models on ASQA is 2-8%. Prior work fixes one piece each: FLARE (adaptive retrieval by prompting), SAIL (always prepends), Toolformer (API calls), RLHF (expensive), self-refine and chain-of-verification (many inference calls).

## Slide 5: Core idea: reflection tokens

This is the central idea. Four token types: Retrieve decides retrieval; three critique tokens: IsRel, IsSup, IsUse. Star = most desirable value. Retrieve, IsRel and IsSup are emitted per segment (one sentence); IsUse once at the end. IsUse is perceived utility: helpfulness regardless of factuality. Because they are vocabulary tokens, judging costs the same as writing a word: no extra classifier head and no separate reward model at inference.

## Slide 6: What the model actually writes

Two real training targets from the paper (Fig. 2 / Table 4). The factual question retrieves; the poem never does: the policy is learned from data. The order matters because the model is autoregressive: earlier tokens condition later ones. Equation: the whole interleaved string is one sequence, so its probability factorises by the chain rule. Each factor is conditioned on everything before it (dots omitted). Passages are inputs supplied by the retriever, so they are not in the product.

## Slide 7: Training: distil a critic, then teach the generator

Two stages. Stage 1: GPT-4 is too expensive to label the whole corpus (150k examples x sentences x passages x 4 aspects) and is not reproducible, so it labels a sample and a 7B critic learns to imitate it (Table 5 agreement numbers). Stage 2: the critic and retriever annotate 145,619 instruction/knowledge examples. When no passage is relevant and supported, a random one is kept, so negative examples remain. The answer text is the original; only tokens and passages are inserted. Then the generator is trained with plain SFT. At inference the critic is gone.

## Slide 8: The two training objectives

Eq. 1: maximum likelihood of the critic's label given input and output: conditional LM used as a classifier. Eq. 2: same NLL objective but over the interleaved sequence of answer tokens and reflection tokens. Token-level form: mask m_i = 0 for retrieved passage tokens (and the prompt). Vocabulary is expanded: new rows in the embedding table and output head. Contrast with RLHF: the critique is computed offline and inserted into data, so training is ordinary SFT.

## Slide 9: Inference, one sentence at a time

Algorithm 1. At every sentence boundary the model emits a Retrieve token; we read its probability and compare with delta. Normalising within the Yes/No group is the restricted softmax: the global normaliser cancels. If retrieving: fetch 5 passages, generate one candidate continuation per passage in parallel (vLLM), each candidate also emits IsRel, IsSup, IsUse. Then keep the best 2 partial answers (beam over sentences, greedy inside a sentence). The 'hard constraints' ablation retrieves only when Yes is the argmax, i.e. delta 0.5, and it hurts PopQA badly.

## Slide 10: Scoring a candidate sentence

Eq. 3 and 4. First term: how likely the sentence is (in the released code, a length-normalised probability). s terms: probability of the best token renormalised within its group. s_Sup is an expected support level (1, 0.5, 0); s_Use is an expected utility on a symmetric scale. Worked example: f(d1) = 0.60 + 0.90 + 0.80 + 0.5*0.57 = 2.585; f(d2) = 0.70 + 0.20 + 0.25 + 0.5*0.325 = 1.31. Set all weights to 0 and the method reduces to likelihood ranking, picking d2.

## Slide 11: Why itworks

Six reasons. The most mathematical one: the optimum of KL-regularised RLHF is pi* proportional to pi_ref times exp(r/beta), so log pi* = log pi_ref + r/beta - log Z. Ranking candidates by likelihood plus a weighted critique reward has the same shape, so Self-RAG approximates decoding from a reward-tilted policy over a finite candidate set; changing w = choosing another reward. Caveat: the paper adds a probability, not a log-probability. Also contrast Lewis-RAG: a soft average weighted by the retriever vs Self-RAG's hard selection weighted by the generator's own judgement.

## Slide 12: Experimental setup

Six tasks in three formats, all zero-shot. Metrics: PopQA and TriviaQA count an answer correct if a gold alias appears in the output. FactScore = fraction of atomic facts supported by Wikipedia (precision, not coverage). ASQA: str-em correctness, MAUVE fluency (distribution similarity to human text), citation recall (are claims entailed by their citations?) and precision (is each citation needed?). Key baseline: Llama2-FT, trained on the same 150k pairs without reflection tokens, isolates the framework from the data.

## Slide 13: Main results

Table 2. Self-RAG is the best non-proprietary model on all tasks. Standard RAG can hurt: ChatGPT drops from 70.1 to 54.7 on PubHealth. Citation precision jumps from 2-5% for open RAG models to 70.3%, above Ret-ChatGPT's 65.1; recall (71.3) is still below ChatGPT's 76.6. Be honest that the abstract's ChatGPT claim is task-dependent. Llama2-FT shows the data alone does not explain the gain. Grounding: only about 2% of Self-RAG's correct answers are absent from its passages, vs 15-20% for baselines.

## Slide 14: Ablations: every piece matters

Ablations use a 7B model trained on 50k examples. Training ablations: removing the critic (always top-1 passage, no tokens, similar to SAIL) drops ASQA from 32.1 to 18.1. Inference ablations: never retrieving drops PopQA by about 21 points but PubHealth by only 0.5. Hard constraints retrieve only when Yes is the argmax, which is too rarely. Always using the top-1 passage (standard RAG) loses ~3.5 points. Dropping IsSup mainly hurts long-form ASQA. Right side: Fig. 3b precision/fluency trade-off, Fig. 3c threshold, Fig. 4 data scaling and a 50-sample human evaluation.

## Slide 15: Critical analysis

Pick two or three limitations to argue in depth. GPT-4 labels at temperature 1, with only 20 hand-checked items per type. The critic agrees with GPT-4 only 80% on IsRel and 73.5% on IsUse, and those errors are distilled into M (Table 6: 'The Lie' example wrongly labelled Relevant + Fully supported). The retriever is never trained, and there is no re-retrieval loop. The scoring weights and the 0.5 / (-1..1) scales are hand-picked. Evaluation caveats: web results for Bio/QA, copied baseline numbers, a 50k model for ablations, no seeds, a 50-sample human evaluation.

## Slide 16: Impact

Context beyond the paper; check details before presenting. CRAG (Corrective RAG, 2024) adds a retrieval evaluator with a web-search fallback, which addresses the frozen-retriever limitation. Adaptive-RAG routes queries by complexity. LangGraph and LlamaIndex ship Self-RAG / CRAG recipes where an LLM is prompted to act as graders, the same roles as IsRel, IsSup and IsUse. Modern assistants treat search as a tool the model chooses to call (a generalised Retrieve token), and RL-trained search agents learn when to search from outcome rewards instead of offline labels.

## Slide 17: Team Contribution

Required by the rubric, presented at the end. Fill in honestly: Equal or Partial, and a score such as 1.0 / 0.8.

## Slide 18: 

Q&A. Likely questions: why mask passages; why a separate critic; what Eq. 3/4 mean; why delta = 0.2; what removing the critic does (ASQA 32.1 to 18.1); is it data or framework (Llama2-FT); biggest limitation; relation to modern LLMs.

