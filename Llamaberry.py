import os
import asyncio
import gradio as gr
from groq import AsyncGroq
import time

# Initialize Groq client
client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

# Define model
model = "llama-3.1-70b-versatile"

# Initial system prompt (regular Chain of Thought)
initial_system_prompt = """You are an AI assistant capable of detailed, step-by-step thinking. When presented with a question or problem, break down your thought process into clear, logical steps. For each step, explain your reasoning. Conclude with a final answer. Use the following markdown structure:

## Reasoning
1. [First step]
   **Explanation:** [Detailed explanation of this step]
2. [Second step]
   **Explanation:** [Detailed explanation of this step]
...

## Answer
[Final answer]

Be comprehensive and show your reasoning clearly."""

# Followup system prompt
followup_system_prompt = """You are an AI assistant tasked with analyzing and improving upon previous problem-solving steps. Review the original query and the previous turns of reasoning, then provide a new perspective or deeper analysis. Use the following markdown structure:

## Critique
[Provide a brief critique of the previous reasoning, highlighting its strengths and potential weaknesses]

## New Reasoning
1. [First step of new or refined approach]
   **Explanation:** [Detailed explanation of this step, referencing the previous reasoning if relevant]
2. [Second step of new or refined approach]
   **Explanation:** [Explanation of how this step builds upon or differs from the previous thinking]
...

## Updated Answer
[Updated answer based on this new analysis]

Be critical yet constructive, and strive to provide new insights or improvements."""

# Synthesis prompt
synthesis_prompt = """You are an AI assistant tasked with synthesizing multiple turns of reasoning into a final, comprehensive answer. You will be presented with three different turns of reasoning for solving a problem. Your task is to:

1. Analyze each turn, considering its strengths and weaknesses.
2. Compare and contrast the different methods.
3. Synthesize the insights from all turns into a final, well-reasoned answer.
4. Provide a concise, clear final answer that a general audience can understand.

Use the following markdown structure:

## Analysis of Turns
[Provide a brief analysis of each turn of reasoning]

## Comparison
[Compare and contrast the turns, highlighting key differences and similarities]

## Final Reasoning
[Provide a final, synthesized reasoning process that combines the best insights from all turns]

## Comprehensive Final Answer
[Comprehensive final answer]

## Concise Answer
[A brief, clear, and easily understandable version of the final answer, suitable for a general audience. This should be no more than 2-3 sentences.]

Be thorough in your analysis and clear in your reasoning process."""


async def call_llm(messages: list,
                   temperature: float = 0.7,
                   max_tokens: int = 8000) -> str:
    """Call the Groq API."""
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


async def generate_turn(query: str, previous_turns: list = None) -> str:
    """Generate a single turn of reasoning, considering previous turns if available."""
    is_first_turn = previous_turns is None or len(previous_turns) == 0
    if is_first_turn:
        messages = [{
            "role": "system",
            "content": initial_system_prompt
        }, {
            "role": "user",
            "content": query
        }]
    else:
        previous_content = "\n\n".join(previous_turns)
        messages = [{
            "role": "system",
            "content": followup_system_prompt
        }, {
            "role":
            "user",
            "content":
            f"Original Query: {query}\n\nPrevious Turns:\n{previous_content}\n\nProvide the next turn of reasoning."
        }]

    return await call_llm(messages)


async def synthesize_turns(query: str, turns: list) -> str:
    """Synthesize multiple turns of reasoning into a final answer."""
    turns_text = "\n\n".join(
        [f"Turn {i+1}:\n{turn}" for i, turn in enumerate(turns)])
    messages = [{
        "role": "system",
        "content": synthesis_prompt
    }, {
        "role":
        "user",
        "content":
        f"Original Query: {query}\n\nTurns of Reasoning:\n{turns_text}"
    }]
    return await call_llm(messages)


async def full_cot_reasoning(query: str) -> tuple:
    """Perform full Chain of Thought reasoning with multiple turns."""
    start_time = time.time()
    turns = []
    turn_times = []
    full_output = f"# Chain of Thought Reasoning\n\n## Original Query\n{query}\n\n"

    for i in range(3):  # Generate 3 turns of reasoning
        turn_start = time.time()
        turn = await generate_turn(query, turns)
        turns.append(turn)
        turn_times.append(time.time() - turn_start)
        full_output += f"## Turn {i+1}\n{turn}\n\n"

    mid_time = time.time()
    synthesis = await synthesize_turns(query, turns)
    full_output += f"## Synthesis\n{synthesis}\n\n"
    end_time = time.time()

    timing = {
        'turn_times': turn_times,
        'total_turns_time': mid_time - start_time,
        'synthesis_time': end_time - mid_time,
        'total_time': end_time - start_time
    }

    full_output += f"## Timing Information\n"
    full_output += f"- Turn 1 Time: {timing['turn_times'][0]:.2f}s\n"
    full_output += f"- Turn 2 Time: {timing['turn_times'][1]:.2f}s\n"
    full_output += f"- Turn 3 Time: {timing['turn_times'][2]:.2f}s\n"
    full_output += f"- Total Turns Time: {timing['total_turns_time']:.2f}s\n"
    full_output += f"- Synthesis Time: {timing['synthesis_time']:.2f}s\n"
    full_output += f"- Total Time: {timing['total_time']:.2f}s\n"

    return full_output


def gradio_interface(query: str) -> str:
    """Gradio interface function."""
    return asyncio.run(full_cot_reasoning(query))


# Create Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[gr.Textbox(label="Enter your question or problem")],
    outputs=[gr.Markdown(label="Chain of Thought Reasoning")],
    title="Multi-Turn Chain of Thought Reasoning with Final Synthesis",
    description=
    "Enter a question or problem to see multiple turns of reasoning, followed by a final synthesized answer."
)

# Launch the interface
iface.launch(share=True)
