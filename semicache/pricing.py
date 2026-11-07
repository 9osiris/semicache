"""tiny price table so stats can show dollars saved. unknown models cost 0."""

# dollars per 1m tokens, (input, output)
PRICES = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-1": (15.00, 75.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "llama-3.3-70b-instruct": (0.35, 0.40),
    "qwen3-32b": (0.10, 0.30),
    "deepseek-chat": (0.27, 1.10),
    "grok-4": (3.00, 15.00),
}


def price_for(model):
    """(input, output) $/1m for a model, or None when it is not in the table."""
    if model in PRICES:
        return PRICES[model]
    # strip a provider prefix, openrouter style names still match the base
    base = model.split("/")[-1]
    return PRICES.get(base)


def cost_usd(model, prompt_tokens, completion_tokens):
    """dollars a call would have cost. 0.0 and unknown=True for unlisted models."""
    price = price_for(model)
    if price is None:
        return 0.0, True
    total = (prompt_tokens * price[0] + completion_tokens * price[1]) / 1_000_000
    return total, False
