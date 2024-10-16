## This is POC for a caseworker using langchain 
- This is heavly inspired by https://github.com/DecisionsDev/rule-based-llms/tree/main/doc 
- The rules engine is supposed to overwrite the knowledge in the llm-agent and the RAG.

### The agent has two sources
- [x] The agent can query a rules engine
- [x] The agent can use a vector document store (RAG)

### The RAG data
- [folketrygd loven](https://lovdata.no/dokument/NL/lov/1997-02-28-19/*#&#x2a;)
    - Get entire page and printed the html as pdf using Chrome. Easy way to manually crawl.

### The rules engine
- Simple rules engine mocked in Python
    - Firm specific rule: can retire at 55
    - Firm specific rule: more than 15 years tenure grants two weeks extra vacation

### Testing
- Added rule which checks if the user is eligable for retirement. Hopefully, this mocks a rules engine.
    - [x] Test query: "kan person med kundenummer 0202196000002 pensjonere seg?"
        - Can retire due to rule engine requires age of 55
    - [x] Test query: "kan person med kundenummer 0202198000001 pensjonere seg?"
        - Can not retire due to both law and rule engine requires a higher age
    - [x] Test query: "hvor lenge kan 0202196000002 ta ferie?"
        - Bjorn Bjorn (kundenummer 0202196000002) can take up to 49 days of vacation.
    - [x] Test query: "hvor lenge kan 0202196000001 ta ferie?"
        - Anna Anna (kundenummer 0202196000001) can take up to 35 days of vacation.

## Setup Instructions
Install pyproject.toml

`poetry install`

Activate a shell

`poetry shell`

Run scripts while in shell 

`python main.py`

Make sure to update `.env` file with API keys. A template key is given in `.env.example`