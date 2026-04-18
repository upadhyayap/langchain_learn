# LangChain Notes

## Core Concepts

- **PromptTemplate** — template for constructing prompts
- **Chat object** — interface for chat-based LLMs
- **Chain** — compose prompt + model + output parser into a pipeline
- LangChain has built-in output parsers

---

## Agents

LangChain agents are autonomous AI systems that perform complex tasks by breaking them down into smaller steps. They use a combination of tools and an LLM to decide which actions to take.

**Agents can:**
1. Plan and execute multi-step tasks
2. Use tools to interact with external systems
3. Make decisions based on context and available tools
4. Adapt their approach based on intermediate results

You can chain agents and hook them up with the LLM to perform complex tasks.

---

## Tools

LangChain tools are functions that agents use to interact with external systems or perform specific tasks.

**Types:**
1. Built-in tools (search, calculator, web browser)
2. Custom tools created by wrapping your own functions
3. Tools that integrate with external APIs and services

Tools are essential components that enable agents to perform real-world actions beyond just generating text.

---

## Output Parsers

LangChain output parsers structure and format raw LLM text output into usable formats.

**Capabilities:**
1. Convert text into structured data types (JSON, lists, custom objects)
2. Extract specific information from LLM responses
3. Format outputs according to predefined schemas
4. Handle parsing errors and provide fallback options

**Common parsers:** `StrOutputParser`, `JSONOutputParser`, `PydanticOutputParser`

---

## Prompts

### Prompt Components

A prompt can be broken down into 4 components:

- **Instructions** — which task needs to be performed; sets the stage for the model response
- **Context** — additional information to fine-tune the instructions
- **Input data** — data that the model needs to process
- **Output indicators** — expected format or structure of the response

### Prompting Techniques

#### Zero-Shot Prompting
Provide the model with a task without any examples. The model relies solely on its pre-trained knowledge.

> Example: *"Translate this English text to French: 'Hello, how are you?'"*

Simple but may not always yield the most accurate results compared to few-shot prompting.

#### Few-Shot Prompting
Provide a few examples of desired input-output pairs before asking the model to perform the task. Examples help the model understand the expected format and style.

> Example (sentiment classification):
> - "I love this movie" → positive
> - "This was terrible" → negative
> - "Classify: [new text]"

Typically yields better results than zero-shot prompting.

#### Chain of Thought Prompting
Ask the model to break down its reasoning step by step before arriving at the final answer. Useful for mathematical problems, logical reasoning, and complex decision-making.

> Instead of: *"What is 25% of 80?"*
> Use: *"Let's solve step by step: First, what is 25% as a decimal? Then, how do we multiply that by 80? Show your work."*

#### ReAct Prompting (Reason + Act)
Combines reasoning and action-taking in a step-by-step format. Alternates between **thinking** (reasoning) and **doing** (acting).

**Example:**

> **Question:** "What is the weather in New York and should I bring an umbrella?"
>
> **Thought:** I need to check the current weather in New York to determine if an umbrella is needed.
> **Action:** Search for "current weather New York"
> **Observation:** Current temperature is 72°F with 60% chance of rain
> **Thought:** Since there's a high chance of rain, an umbrella would be useful
> **Action:** Recommend bringing an umbrella
> **Final Answer:** Yes, you should bring an umbrella to New York today as there's a 60% chance of rain despite the mild temperature of 72°F.

This structured approach helps models make more informed decisions by explicitly separating reasoning from actions.

---

## MCP Server

A **Model Control Protocol (MCP)** server is a specialized server that manages and coordinates interactions between language models and various tools or services. It acts as a middleware layer that:

1. Handles model requests and responses
2. Manages tool execution and integration
3. Provides a standardized interface for model-tool communication
4. Enables secure and controlled access to external systems

MCP servers are crucial for building robust AI applications that require reliable model-tool interactions.

**Implementation:**
- Use **FastMCP** to build the MCP server
- Read about the difference between **stdio** and **SSE** in MCP servers

**LangChain** has a `MultiServerMCPClient` that can connect to multiple MCP servers simultaneously.
