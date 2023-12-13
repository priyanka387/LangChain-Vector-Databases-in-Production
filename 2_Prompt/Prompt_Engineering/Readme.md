# **Intro to Prompt Engineering: Tips and Tricks**

## **Introduction**
Prompt engineering is a relatively new discipline that involves developing and optimizing prompts to use language models for various applications and research topics efficiently. It helps to understand the capabilities and limitations of LLMs better and is essential for many NLP tasks. We will provide practical examples to demonstrate the difference between good and bad prompts, helping you to understand the nuances of prompt engineering better.

## **Role Prompting**
Role prompting involves asking the LLM to assume a specific role or identity before performing a given task, such as acting as a copywriter. This can help guide the model's response by providing a context or perspective for the task. To work with role prompts, you could iteratively:

- Specify the role in your prompt, e.g., "As a copywriter, create some attention-grabbing taglines for AWS services."
- Use the prompt to generate an output from an LLM.
- Analyze the generated response and, if necessary, refine the prompt for better results.

## Examples:

In this example, the LLM is asked to act as a futuristic robot band conductor and suggest a song title related to the given theme and year. (A reminder to set your OpenAI API key in your environment variables using the OPENAI_API_KEY key) Remember to install the required packages with the following command: pip install langchain==0.0.208 deeplake openai tiktoken.

```
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

# Initialize LLM
llm = OpenAI(model_name="text-davinci-003", temperature=0)

# Prompt 1
template_question = """What is the name of the famous scientist who developed the theory of general relativity?
Answer: """
prompt_question = PromptTemplate(template=template_question, input_variables=[])

# Prompt 2
template_fact = """Provide a brief description of {scientist}'s theory of general relativity.
Answer: """
prompt_fact = PromptTemplate(input_variables=["scientist"], template=template_fact)

# Create the LLMChain for the first prompt
chain_question = LLMChain(llm=llm, prompt=prompt_question)

# Run the LLMChain for the first prompt with an empty dictionary
response_question = chain_question.run({})

# Extract the scientist's name from the response
scientist = response_question.strip()

# Create the LLMChain for the second prompt
chain_fact = LLMChain(llm=llm, prompt=prompt_fact)

# Input data for the second prompt
input_data = {"scientist": scientist}

# Run the LLMChain for the second prompt
response_fact = chain_fact.run(input_data)

print("Scientist:", scientist)
print("Fact:", response_fact)
The sample code.
Copy
Scientist: Albert Einstein
Fact: 
Albert Einstein's theory of general relativity is a theory of gravitation that states that the gravitational force between two objects is a result of the curvature of spacetime caused by the presence of mass and energy. It explains the phenomenon of gravity as a result of the warping of space and time by matter and energy.
```