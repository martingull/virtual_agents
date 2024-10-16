## Create a legal agent using langchain

The pipeline follows RAG + Agent setup. Components are:
 - Compendium of legal documents and "how to ace Bar exam" as domain documents
 - BERT Legal for embeddings
 - ChatGPT 4.o as a llm agent 

 Comments:
- [x] Seems to work. Being tested on questions from the BAR exam 
- [x] Needs more documents. Using compendium of EU law as dummy data. Making the legal agent an EU lawyer. Seems to work. 
- [x] Found documents on Pirate Bay "Bar Exam Preparation" to be added to the RAG stage. 
