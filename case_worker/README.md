## This is POC for a caseworker using langchain 
- This is heavly inspired by https://github.com/DecisionsDev/rule-based-llms/tree/main/doc 

### The agent has two sources
- [] The agent can query a rules engine
- [] The agent can use a vector document store (RAG)

### The RAG data
- [folketrygd loven](https://lovdata.no/dokument/NL/lov/1997-02-28-19/*#&#x2a;)

### The rule system
- [no clue atm](d)

### Testing
- [] Added rule which checks if the user is eligable for retirement. Hopefully, this mocks a rules engine.
    - Test query: "kan man pensjonere seg ved en alder av 60"
