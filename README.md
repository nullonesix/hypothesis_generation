# hypothesis_generation
Generate scientific hypotheses with the most information gain for the language model.

from https://medium.com/@nullonesix/motivated-question-asking-ai-38876bdfe14b:

Language Models That Ask Questions

ChatGPT asks questions but they’re human prompted. It would be better if ChatGPT went off on its own, asked a bunch of questions to itself, and then came back to us with its best question.

What’s the best question?

The one whose response produces the largest gradients.

What’s the pseudocode?

1. Sample a statement X from ChatGPT.
2. Set X and ¬X to be the expected outputs.
3. Compute the mean of the absolute value of the gradients M.
4. If it’s the biggest M seen so far, save it and the associated X.
5. Repeat steps 1–4 for an hour.
6. Print X and await human yes/no response.
7. Learn on the version of X or ¬X that’s true.
8. Repeat steps 1–7 forever.

End notes:

this article came about with the realization that we don’t have many examples of AI that generally think for themselves
usually either the AI is thinking about new data coming in (English text from the internet when training language models) or otherwise performing some predefined algorithm on old data (MCTS search on the current chess position)
the question arose as to how to define thinking, and I took the scientific approach to be an approximation, i.e. hypothesis generation and subsequent testing of worthwhile hypotheses
here the AI is generating its own data in a hypothesis generating process, selecting the best hypothesis to test based on its informativeness, and then testing that hypothesis by asking a human
    this draws attention to two followup problems:
    1. how to make the hypothesis generating process more efficient
    2. how to automate hypothesis testing
