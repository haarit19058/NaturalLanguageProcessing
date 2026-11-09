This is the mathematical foundation of dense retrieval.

Important: The model used for retrieving information is called a retriever. The model generating the final answer is called a generator.

They are different components.

The LLM generates its answer conditioned on the question and retrieved passages.

Mathematically:
\[
P_\phi(y\mid x,\mathcal D_K)
\]

where:
- \(x\): User's question.
- \(\mathcal D_K\): Retrieved passages.
- \(y\): Generated response.
- \(\phi\): Generator parameters.
A simplified RAG system can therefore be represented as:
\[
x
\overset{\text{Retriever}}{\longrightarrow}
\mathcal D_K
\overset{\text{Generator}}{\longrightarrow}
y
\]

This is the traditional RAG pipeline.


