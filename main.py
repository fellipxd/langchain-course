import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain!")
    information = """
John McCarthy is one of the "founding fathers" of artificial intelligence, together with Alan Turing, Marvin Minsky, Allen Newell, and Herbert A. Simon. McCarthy, Minsky, Nathaniel Rochester and Claude E. Shannon coined the term "artificial intelligence" in a proposal that they wrote for the famous Dartmouth conference in Summer 1956. This conference started AI as a field.[9][15] (Minsky later joined McCarthy at MIT in 1959.)

In 1958, he proposed the advice taker, which inspired later work on question-answering and logic programming.

In the late 1950s, McCarthy discovered that primitive recursive functions could be extended to compute with symbolic expressions, producing the Lisp programming language.[16] That functional programming seminal paper also introduced the lambda notation borrowed from the syntax of lambda calculus in which later dialects like Scheme based its semantics. Lisp soon became the programming language of choice for AI applications after its publication in 1960.

In 1958, McCarthy served on an Association for Computing Machinery ad hoc committee on Languages that became part of the committee that designed ALGOL 60. In August 1959 he proposed the use of recursion and conditional expressions, which became part of ALGOL.[17] He then became involved with developing international standards in programming and informatics, as a member of the International Federation for Information Processing (IFIP) Working Group 2.1 on Algorithmic Languages and Calculi,[18] which specified, maintains, and supports ALGOL 60 and ALGOL 68.[19]

Around 1959, he invented so-called "garbage collection" methods, a kind of automatic memory management, to solve problems in Lisp.[16][20]

During his time at MIT, he helped motivate the creation of Project MAC, and while at Stanford University, he helped establish the Stanford AI Laboratory, for many years a friendly rival to Project MAC.

McCarthy was instrumental in the creation of three of the very earliest time-sharing systems (Compatible Time-Sharing System, BBN Time-Sharing System, and Dartmouth Time-Sharing System). His colleague Lester Earnest told the Los Angeles Times:

The Internet would not have happened nearly as soon as it did except for the fact that John initiated the development of time-sharing systems. We keep inventing new names for time-sharing. It came to be called servers ... Now we call it cloud computing. That is still just time-sharing. John started it.[9]

— Elaine Woo
In 1961, he was perhaps the first to suggest publicly the idea of utility computing, in a speech given to celebrate MIT's centennial: that computer time-sharing technology might result in a future in which computing power and even specific applications could be sold through the utility business model (like water or electricity).[21][22] This idea of a computer or information utility was very popular during the late 1960s, but had faded by the mid-1990s. However, since 2000, the idea has resurfaced in new forms (see application service provider, grid computing, and cloud computing).

In 1966, McCarthy and his team at Stanford wrote a computer program used to play a series of chess games with counterparts in the Soviet Union; McCarthy's team lost two games and drew two games (see Kotok-McCarthy).

From 1978 to 1986, McCarthy developed the circumscription method of non-monotonic reasoning.

In 1982, he seems to have originated the idea of the space fountain, a type of tower extending into space and kept vertical by the outward force of a stream of pellets propelled from Earth along a sort of conveyor belt which returns the pellets to Earth. Payloads would ride the conveyor belt upward.[23]
    """

    summary_template = """
        {information} I want you to create:
        given the following information:
        1. A short summary
        2. A list of 5 key points
        3. A list of 5 questions that a curious reader might have after reading the summary
        4. A list of 5 answers to the questions
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    # llm = ChatOllama(model="gemma3:270m", temperature=0)
    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
