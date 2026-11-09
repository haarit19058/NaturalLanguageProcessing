# Self-RAG presentation: speaker notes

## Slide 1: Self-RAG

Self-RAG (Asai et al., UW / AI2 / IBM, ICLR 2024). One-line pitch: a single LM that decides when to retrieve, judges what it retrieved and grades its own sentences, using special tokens in its vocabulary; the grades are then used to steer decoding.

## Slide 2: Roadmap

Structure follows the CS613 rubric: problem and motivation, core contribution, technical method, experiments, critical analysis, impact. Backup slides at the end hold extra detail for Q&A.

## Slide 4: Language models are probability distributions

Everything later is next-token prediction. Two details matter: the masked loss (Self-RAG masks retrieved passages), and the fact that critique tokens will be predicted through the same softmax as words.

## Slide 5: Where does an LM keep its knowledge?

Parametric vs non-parametric memory. PopQA is from the same group (Mallen et al. 2023). Retrieval triples accuracy on long-tail questions, which motivates RAG; the rest of Part 1 shows why plain RAG is not enough.

## Slide 6: Retrieval: sparse vs dense

BM25 vs dense retrieval. InfoNCE is cross-entropy over one positive and m negatives. Important for critique: because R is frozen, Self-RAG can detect bad passages (IsRel) but cannot improve retrieval.

## Slide 7: Retrieval-augmented generation: three formulations

Lewis-RAG treats the passage as a latent variable and marginalises (sequence- or token-level). In-context RALM just concatenates. All three retrieve always and once; none checks whether the output follows the evidence. Contrast for later: Self-RAG does hard selection weighted by the LM's own judgment, not soft mixing weighted by the retriever.

## Slide 8: Why plain RAG falls short

Three failure modes from the introduction with evidence from Table 2. The formal condition shows the retrieval decision should be input- and prefix-dependent.

## Slide 9: Prior work (1/2): when and what to retrieve

Related work, retrieval side. The common gap: retrieval is either always-on or triggered heuristically, and nothing judges relevance and support jointly inside one model.

## Slide 10: Prior work (2/2): critics, control tokens, refinement

Related work, critique side. Self-RAG takes the 'learned critic' idea from RLHF, the 'special token' idea from control-token work, and makes the tokens self-evaluations used at inference time.

## Slide 11: What the authors set out to achieve

Six design goals that map onto the components: Retrieve token, IsRel, IsSup, selective retrieval, weights and threshold, SFT on an annotated corpus.

## Slide 13: Conventional RAG vs Self-RAG (paper Fig. 1)

Recreation of Figure 1. Left: standard RAG always retrieves K and never revisits quality. Right: Self-RAG decides to retrieve, processes passages in parallel, judges relevance and support, and selects the best continuation; the essay prompt does not retrieve at all.

## Slide 14: From the paper: Figure 1

Original Figure 1. Walk through steps 1-3 on the right; point at the ranking '1 > 3 > 2' produced by the critique tokens.

## Slide 15: Reflection tokens (paper Table 1)

Retrieve is the retrieval token; IsRel, IsSup, IsUse are critique tokens. The bold/best value is the most desirable one. Retrieve, IsRel and IsSup are per segment; IsUse once per output.

## Slide 16: From the paper: Table 1 and Algorithm 1

Original Table 1 (token types; bold = most desirable value) and Algorithm 1 (inference). M_gen in lines 9-10 is the same generator.

## Slide 17: Token definitions in detail (App. A.1)

Precise definitions from Appendix A.1. Note IsUse deliberately ignores factuality, so it is down-weighted (0.5) at inference.

## Slide 18: What is new

Five novelty claims. Be precise in Q&A: adaptive retrieval existed (FLARE, Toolformer) and critics existed (RLHF); the novelty is learning both as tokens in one model, offline, and using them for controllable decoding.

## Slide 19: Self-RAG vs other RAG designs

Side-by-side comparison. Highlighted column is Self-RAG. Note the training row: Lewis-RAG trains the retriever, Self-RAG does not.

## Slide 21: Problem formalisation (§3.1)

Formal model from §3.1 and Algorithm 1. The retrieval call is the only step outside the LM. Everything else is sampled from p_M.

## Slide 22: One token stream, so the chain rule applies

Because everything is one autoregressive stream, the probability factorises by the chain rule. The token order is a deliberate design choice because earlier tokens condition later ones.

## Slide 23: What the model writes (paper Fig. 2, Table 4)

Real training targets. The factual question retrieves; the poem never does; the delegation example mixes No and Yes, and keeps an Irrelevant passage as a negative example. Passages are wrapped in <p>…</p> (paragraph tokens).

## Slide 24: From the paper: Figure 2 (training examples)

Original Figure 2: how the critic turns an (input, output) pair into an augmented training sequence.

## Slide 25: Vocabulary expansion and the restricted softmax

Vocabulary expansion: two matrices grow. The restricted-softmax identity is used for the retrieval threshold and every critique score. Token strings are from the released checkpoints (selfrag/selfrag_llama2_7b).

## Slide 26: Training overview (Algorithm 2)

Algorithm 2. Keep the distillation chain GPT-4 → critic → generator in mind; it explains most design choices and the main limitation.

## Slide 27: Stage 1a: critic labels from GPT-4

Data collection for the critic. Counts and agreement numbers from Appendix A.1. Lower utility agreement is mostly 4-vs-5 and 1-vs-2 confusion.

## Slide 28: Stage 1b: training the critic (Eq. 1)

Critic objective and accuracy. Point out in Q&A that errors here become training targets for the generator (error cascade).

## Slide 29: Stage 2a: annotating the corpus (Alg. 3)

Algorithm 3 in detail (Appendix A.2). Critical subtlety: the original answer is unchanged, so the generator learns to annotate good answers, not new content.

## Slide 30: Training data: 145,619 examples (App. Table 3)

Appendix Table 3. Category totals are computed from the table (92,008 + 53,611 = 145,619). Some evaluation-adjacent sets appear in training (ASQA train, ARC-Easy), which is worth noting.

## Slide 31: Stage 2b: training the generator (Eq. 2)

Generator objective, written at token level with the mask. Versus Eq. 1, the generator writes and grades. Masking passages is a likely exam question: they are not model output.

## Slide 32: Why not RLHF? Cost, stability, control

Contrast with PPO. Self-RAG computes critique offline and trains with SFT. The trade-off is that the generator is not optimised against the reward; improvement comes from test-time selection.

## Slide 33: Inference: Algorithm 1

Algorithm 1, lightly paraphrased. The next three slides formalise lines 2 and 8: the threshold, the critique scores and the segment score.

## Slide 34: Adaptive retrieval threshold (App. A.3)

Threshold rule from Appendix A.3. The worked example (exercise E1 in the guide) shows why the soft threshold retrieves more than the argmax rule.

## Slide 35: Turning critique tokens into scores

Exact score definitions from Appendix A.3. The expected-value view is my reformulation: each score is the expected value of a per-token value under the group-restricted distribution. It explains the 0.5 credit for 'Partially' and the symmetric utility scale.

## Slide 36: Segment score and beam search (Eqs. 3–4)

Eq. 3 and 4. The paper states p(y_t|…) in Eq. 3; the length-normalised form is how the released code implements it (to our reading). Beam search operates over sentences, not tokens.

## Slide 37: Worked example: why the less fluent candidate wins

Hand-worked example (guide §11.5). f(d1) = 0.60 + 0.90 + 0.80 + 0.5×0.57 = 2.585; f(d2) = 0.70 + 0.20 + 0.25 + 0.5×0.325 = 1.3125. Weight-flip analysis is exercise E4.

## Slide 38: Inference cost

Cost estimate is my own; the paper does not report wall-clock latency, which is a fair criticism.

## Slide 39: Why itworks

Six reasons it works. Generate-then-verify is the same principle as best-of-N and reranking.

## Slide 40: A closer look: test-time reward tilting

Derivation: the KL-regularised RL objective has a closed-form optimum proportional to the reference policy times exp(r/β). Self-RAG's ranking has the same shape with the critique sum as reward. This is an interpretation, not a claim from the paper.

## Slide 42: Tasks and datasets (§4.1), all zero-shot

Six tasks across three formats. All zero-shot with task instructions (App. Table 5). TriviaQA uses the Min et al. 2019 validation/test split because the open test set is not public.

## Slide 43: Metrics (1/2): accuracy, match, FactScore

Metric definitions. The FactScore caveat explains why Self-RAG 7B (81.2) beats 13B (80.2): shorter, precisely grounded outputs.

## Slide 44: Metrics (2/2): ASQA correctness, fluency, citations

ASQA metrics. MAUVE formula from Pillutla et al. 2021. Citation precision and recall follow Gao et al. 2023 (ALCE).

## Slide 45: Baselines (§4.2) and what each one controls for

The most important controlled baseline is Llama2-FT (highlighted): same base model, same 150k examples, retrieval at test time only.

## Slide 46: Evaluation settings and fairness

Implementation details that matter for fair comparison. Note the web-search augmentation for Bio/QA: the text states shared retrieval explicitly only for ASQA.

## Slide 47: Table 2 (1/2): vs proprietary and no-retrieval models

First half of Table 2. Pink cells show where ChatGPT or Ret-ChatGPT is better: TriviaQA, ARC, ASQA em/rouge/MAUVE and citation recall. Self-RAG beats ChatGPT on PopQA, PubHealth, Bio and citation precision.

## Slide 48: Table 2 (2/2): vs retrieval-augmented baselines

Second half of Table 2. Self-RAG is the best non-proprietary model on every task; on ASQA str-em Alpaca-13B+RAG (34.8) is higher than Self-RAG 13B (31.7), so 'best on all tasks' refers to the main metric per task in the paper's bolding. Mention this nuance if asked.

## Slide 49: From the paper: Table 2

Original Table 2 for reference; the two rebuilt tables carry the same numbers with highlighting.

## Slide 50: Main results at a glance

Visual summary of Table 2 against the strongest comparators.

## Slide 51: Reading Table 2: five takeaways

Five takeaways with numbers. Be honest that the ChatGPT comparison is task-dependent.

## Slide 52: Do answers come from the evidence? (App. C.1)

Appendix C.1. Self-RAG's correct answers are almost always traceable to evidence. The 'Alpaca 30B' label in the appendix looks like a typo; the figures are 20% / 18% / 15% for the three baselines.

## Slide 53: Ablations (Fig. 3a): every component matters

Fig. 3a. Training ablations: no retriever, no critic. Inference ablations: no retrieval, hard constraints, top-1, remove IsSup. Largest effects: no critic on ASQA (−14.0), never retrieve on PopQA (−20.8), hard constraint on PopQA (−17.2).

## Slide 54: From the paper: Figure 3

Original Figure 3. The next two slides re-plot (b) and (c) with approximate values read from this figure.

## Slide 55: Test-time customisation (Fig. 3b, ASQA)

Fig. 3b. Precision rises slightly, MAUVE drops sharply. Consistent with Liu et al. 2023a: longer, more fluent text contains more unsupported claims.

## Slide 56: Efficiency vs accuracy: the threshold δ (Fig. 3c)

Fig. 3c. Retrieval frequency changes dramatically with δ; the accuracy cost of retrieving less is small for PubHealth and large for PopQA.

## Slide 57: Training-data scale (Fig. 4a–c)

Fig. 4a-c. Upward trends on PopQA and ASQA; PubHealth is flat. The authors limited training to 150k.

## Slide 58: From the paper: Figure 4

Original Figure 4. The 150k ASQA precision point in the figure (~70) is a little higher than the 7B Table 2 value (66.9).

## Slide 59: Human evaluation (Fig. 4d, App. Table 6)

Small human study (50 samples). Show the failure case: the model can be confidently wrong about its own support label.

## Slide 60: Qualitative examples (App. Table 7, Self-RAG 13B)

Three examples from Appendix Table 7 showing support, contradiction and partial support labels.

## Slide 62: Strengths

Six strengths. Lead with simplicity and controllability; they are what made the paper influential.

## Slide 63: Limitations (1/3): supervision

Supervision limitations. The two-step distillation is the most defensible 'biggest limitation' answer.

## Slide 64: Limitations (2/3): method

Method limitations. Good improvement suggestions follow from each: re-retrieval, multi-passage segments, learned or calibrated weights, latency reporting.

## Slide 65: Limitations (3/3): evaluation

Evaluation limitations. Mention them neutrally: the core claims still hold against open baselines.

## Slide 66: Is the evidence sufficient for the claims?

A direct answer to the rubric question 'are the experiments sufficient?'. My verdicts, not the paper's.

## Slide 67: Assumptions and how to improve

Have two improvements ready for Q&A; each is tied to a limitation.

## Slide 69: Timeline

Context beyond the paper; verify details before presenting.

## Slide 70: Follow-up work building on Self-RAG

Follow-up works and how they relate. All are outside the paper; cite carefully.

## Slide 71: Self-RAG and modern LLM systems

Mapping each Self-RAG component to its modern counterpart.

## Slide 72: Key takeaways

Six-line summary of the whole talk.

## Slide 73: Team Contribution

Required by the rubric. Fill in honestly.

## Slide 74: References (1/3)

References: the paper, background, baselines, metrics and follow-up work.

## Slide 75: References (2/3)

References: the paper, background, baselines, metrics and follow-up work.

## Slide 76: References (3/3)

References: the paper, background, baselines, metrics and follow-up work.

## Slide 77: 

Q&A. Backup slides follow.

## Slide 79: Q&A bank (1/2)

Backup: short answers to likely questions.

## Slide 80: Q&A bank (2/2)

Backup: short answers to likely questions.

## Slide 81: GPT-4 prompt scales (App. D, Tables 11–12)

Paraphrased from Appendix D. Useful if asked how the labels are defined.

## Slide 82: Evaluation instructions (App. Table 5)

Instructions used at evaluation time. Self-RAG does not need the citation instruction because citation is built into its output format.

