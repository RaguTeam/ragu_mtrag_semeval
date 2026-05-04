# MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems
https://arxiv.org/abs/2501.03468

Previous multi-turn RAG benchmark limitations:
  1. Except for FaithDial, prior datasets keep the retrieval component fixed.
  2. Except for RAD-Bench and iKAT, prior datasets focus on extractive or short answers (1-2 sentences).
  3. Many existing datasets ignore unanswerable questions - a ripe source of hallucinations.
  4. Most datasets focus on a single domain/topic.
We present MTRAG, a multi-turn RAG benchmark that includes active retrieval (the relevant passages change during the conversation), long-form answers, unanswerable questions, and multiple domains.

# Creation

We take 4 document corpora/domains:

  1. CLAPNQ (2024): a subset of Wikipedia pages.
  2. FiQA (2018): StackExchange posts discussing financial advice.
  3. Govt (novel): some inter-connected web-pages under the .gov and .mil domains.
  4. Cloud (novel): some inter-connected technical documentation pages of a major cloud provider.

We hire paid annotators and ask them to create multi-turn conversations of 6-9 or more turns, allowing them to interact with a live RAG agent (ElasticSearch with passages of 512 tokens with overlap 100 +  Mixtral 8X7b Instruct). For each corpus we also assembled a set of seed questions to help human annotators. Annotators were encouraged to create questions that naturally extended the preceding conversation while varying in answerability types, question types, and multiturn patterns.

For each conversation turn we generate response which is then repaired by the annotator as needed. A response that does not require repair can be considered an indication that the question is not challenging for the LLM. The annotator may also adjust the set of retrieved passages. In MTRAG, the Rouge-L similarity between the original and repaired response is 60.7, indicating significant amount of repair, and conversations contain repairs on 92% of the turns. This simulates real-time conversations, which is missing in prior work.

Each reference answer is written to satisfy the following properties referred as FANC (by first letters):

  1. Faithfulness: The answer is faithful to the passages or earlier turns.
  2. Appropriateness: It is appropriate/relevant to the question.
  3. Naturalness: The answer sounds natural.
  4. Completeness: Includes all information in the passages relevant to the question.

The questions occasionally rely on prior turns in the conversation: on average 1.3 questions per conversation include co-references.

Dimensions:

  1. We classify each question in multi-turn conversations into types (factoid, comparison, explanation, keyword etc.), a question may have multiple types - see table 9.
  2. We classsify each question in a conversation, except the first one, as either follow-up or clarification - see table 10.
  3. A question may be answerable, partially answerable, unanswerable based on the corpora, and non-question (e.g., “Hi", “That’s interesting", “Thank you").

The resulting 126 conversations were then reviewed. Annotators could accept or reject conversations, and repair responses, passage relevance, and question type labels as needed, but not allowed to edit the questions.

This process yielded a benchmark of 110 conversations (29 ClapNQ, 27 FiQA, 28 Govt, 26 Cloud) with an average 7.7 turns per conversation, leading to 842 tasks (task = a question to the assistant + dialog history). All evaluations are performed at the task level.

We also construct a companion benchmark, MTRAG-S, of synthetically generated conversations to help the community analyze the relative advantages of the two types of data. Our attempts to synthetically generate unanswerable questions were not very successful as the model would often create questions with at least a partial answer. Increasing the number of turns
tended to lead to repetitive user questions.

# Evaluation

We do not perform a comprehensive evaluation of all retrievers and generators on MTRAG, but only perform some evaluations just to demonstrate the challenging nature of MTRAG. Also, since we use Elser for retrieval during data creation, there may be some biases towards Elser.

We experimented with several strategies to query the retriever. We report Recall and nDCG metrics. Co-referencing is a challenge, so we try a query rewrite strategy to form an unambiguous, standalone question. For example: "Who is the CEO of Apple Inc.?", "The CEO of Apple Inc. is TIM COOK.", "Its address?" -> "What is the address of Apple Inc?". It consistently outperforms using only the last turn. 

To evaluate retrievers, we send the question, preceding turns, N=5 passages, and instructions. In our case, N=5 performs better than N=3 while still being manageable. We use 3 retrieval settings:

  1. Reference passages, or no passages if unanswerable/conversational.
  2. Reference passages + RAG passages.
  3. RAG passages only.

Evaluation metrics for generation:

  1. RB_alg (reference-based algorithmic): the harmonic mean of Bert-Recall (approximates completeness), Bert-K-Precision (compares model response to the passages) and Rouge-L (approximates faithfulness and completenes). Rouge-L and Bert-Rec are calculated using the response and the reference answer, while Bert-K-Prec uses the response and the passages).
  2. RB_llm (reference-based LLM): an LLM judge inspired by RAD-Bench. We compare the response and the reference answer, supplementing the prompt with the passages and anchor the evaluation on the metrics of faithfulness, appropriateness, and completeness. We take a median of several LLM judges.
  3. RL_F: the reference-less faithfulness LLM judge from RAGAS.
  4. IDK-judge ("I Don’t Know") to detect whether the response has a full or partial answer. Intuitively, words used to indicate not knowing the answer may not match the context. If IDK = yes or A = yes (the question answerability), then we use (IDK = A) as a binary score, otherwise we calculate other metrics (RB_alg, RB_llm, RL_F).

All models score significantly lower than the reference answer, indicating there is still room for improvement in MTRAG for all LLMs. The results degrate as the retrieval settings change: Reference -> Reference + RAG -> RAG. Models experience a dramatic drop in performance on unanswerables, struggling to declare they do not know the answer.  While GPT-4o and Llama 405B score low for unanswerables, they still perform much better than models in other families. In almost all cases the models perform better on first turn. Model performance is similar across domains except for FiQA where the results tend to be lower.

We performed a human evaluation of GPT-4o and Llama 3.1 405B on a subset of the benchmark. We select 5 conversations per domain for a total of 159 evaluation tasks. We ask annotators to measure the quality across the desirable response properties (FANC) and to perform a pair-wise model comparison. The annotators see the dialog, the response and the relevant passages. The annotator agreement was very high. The reference answer is exceedingly preferred by annotators, this highlights the quality of the human-generated reference answers. LLMs still struggle with faithfulness and completeness, where they receive lower scores.

# IMO

The distribution of seed questions may alter much if that who asks does not know the answer in advance. If they do, questions tend to be more factoid and require a single passage to answer.

There is a problem in the benchmark collection process. Since the annotator see only the retrieved passages, then they won't even know that there are other relevant chunks, if the retriever failed to find them during the annotation process. This is a questionable feature of the MTRAG benchmark. Moreover, the missed chunks in the rest of the corpus may contain something contradicting, clarifying etc. For example, if a question is "what did NASA launch in 2006", there may be multiple answers, where onle some of them are retrieved and hence are visible to the annotator. So, overall, we cannot be sure that annotators are able to know the ultimate correct answer, since they do not study the whole corpus and only see only the retrieved docs. Also, we cannot be sure that the retrieved chunks set is complete to evaluate retrieval recall.

In human evaluation, if they measure only the correctness, they could take more conversations, while 5 conversations on each domain is not enough.

The comparison of human vs LLM-as-judge evaluation (fig. 4) could be designed better. They could ask a human to compare two models, and also compare these models with LLM-as-judge, then evaluate the ratio of question-answer pairs where the LLM-as-judge rakning is the same as human ranking. This metrics is more easily analyzed than the Pearson correlation.
