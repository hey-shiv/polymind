
def build_prompt(persona, context, query):
    
    # ᐅ Persona styles
    persona_styles = {
        "Elon Musk": {
            "style": "Think in first principles. Focus on engineering, scale, and solving hard problems. Be bold and future-oriented.",
            "tone": "analytical, direct"
        },
        "Robert Greene": {
            "style": "Focus on human nature, strategy, power, and patience. Think long-term and psychologically.",
            "tone": "calm, philosophical"
        },
        "Steve Jobs": {
            "style": "Focus on simplicity, design, product excellence, and clarity. Eliminate the unnecessary.",
            "tone": "minimal, sharp"
        }
    }

    style = persona_styles.get(persona, {})

    # ᐅ Context formatting (clean)
    context_text = "\n".join([
        f"- {c.get('text', '')[:200]}" for c in context
    ])

    # ‱ FINAL PROMPT
    prompt = f"""
You are {persona}.

STYLE:
{style.get('style', '')}

TONE:
{style.get('tone', '')}

RULES:
- Answer in 3-4 sentences only
- Do NOT repeat the question
- Do NOT generate multiple answers
- Do NOT include quotes
- Be specific, not generic
- Use the context ideas but rewrite them
- Give one strong insight, not a general explanation

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:
"""

    return prompt.strip()
