"""Fixed few-shot chain-of-thought templates and answer extraction.

The 8 GSM8K exemplars are plain arithmetic word problems with worked solutions in
the style of Wei et al. (2022), written here so the template is frozen in code.
Every configuration sees byte-identical prompts.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

GSM8K_EXEMPLARS: List[Dict[str, str]] = [
    {"q": "A bakery sold 24 muffins in the morning and 18 in the afternoon. Each muffin costs $3. How much money did the bakery make?",
     "a": "The bakery sold 24 + 18 = 42 muffins. At $3 each that is 42 * 3 = 126 dollars. The answer is 126."},
    {"q": "Priya has 5 boxes with 12 pencils each. She gives 17 pencils to her friends. How many pencils does she have left?",
     "a": "She starts with 5 * 12 = 60 pencils. After giving away 17 she has 60 - 17 = 43 pencils. The answer is 43."},
    {"q": "A train travels 60 miles per hour for 2 hours and then 45 miles per hour for 3 hours. How far does it travel in total?",
     "a": "The first part is 60 * 2 = 120 miles. The second part is 45 * 3 = 135 miles. In total it travels 120 + 135 = 255 miles. The answer is 255."},
    {"q": "Tom reads 15 pages a day. A book has 210 pages. He has already read for 6 days. How many more days does he need to finish the book?",
     "a": "In 6 days Tom read 15 * 6 = 90 pages. He has 210 - 90 = 120 pages left. At 15 pages a day that takes 120 / 15 = 8 days. The answer is 8."},
    {"q": "A farmer has 48 apples. He sells a quarter of them and then eats 3. How many apples does he have now?",
     "a": "A quarter of 48 is 48 / 4 = 12, so he sells 12 and has 48 - 12 = 36 left. After eating 3 he has 36 - 3 = 33 apples. The answer is 33."},
    {"q": "A shirt costs $25. During a sale the price is reduced by 20%. Maya buys 3 shirts on sale. How much does she pay?",
     "a": "20% of 25 is 25 * 0.2 = 5, so the sale price is 25 - 5 = 20 dollars. Three shirts cost 3 * 20 = 60 dollars. The answer is 60."},
    {"q": "There are 8 rows of chairs with 14 chairs in each row. 23 chairs are broken and removed. How many chairs remain?",
     "a": "There are 8 * 14 = 112 chairs. Removing 23 leaves 112 - 23 = 89 chairs. The answer is 89."},
    {"q": "Leo earns $12 an hour and works 7 hours on Saturday and 5 hours on Sunday. He spends $40 on a game. How much money does he have left?",
     "a": "He works 7 + 5 = 12 hours and earns 12 * 12 = 144 dollars. After the game he has 144 - 40 = 104 dollars. The answer is 104."},
]

GSM8K_SYSTEM = "You solve math word problems step by step and finish with 'The answer is N.'"


def gsm8k_prompt(question: str) -> str:
    parts = [f"Question: {e['q']}\nAnswer: {e['a']}" for e in GSM8K_EXEMPLARS]
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


def math500_prompt(question: str) -> str:
    return f"Solve the following problem step by step. Finish with 'The answer is <answer>.'\n\nProblem: {question}\nSolution:"


def arc_prompt(question: str, choices: Dict[str, str]) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in choices.items())
    return f"Answer the multiple-choice question. Think briefly, then finish with 'The answer is <letter>.'\n\nQuestion: {question}\n{opts}\nAnswer:"


def build_prompt(task: str, problem: Dict) -> str:
    if task == "gsm8k":
        return gsm8k_prompt(problem["question"])
    if task == "math500":
        return math500_prompt(problem["question"])
    if task == "arc":
        return arc_prompt(problem["question"], problem["choices"])
    raise ValueError(task)


def chat_wrap(tokenizer, prompt: str, system: Optional[str] = GSM8K_SYSTEM) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


_ANS = re.compile(r"[Tt]he answer is[:\s]*\$?\s*(-?[\d,]*\.?\d+|[A-E])\b")
_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_answer(text: str, task: str = "gsm8k") -> Optional[str]:
    """The number (or letter for ARC) after 'The answer is'; otherwise the last number in the text."""
    from .data import normalise_number

    m = _ANS.findall(text)
    if m:
        ans = m[-1]
        return ans.upper() if task == "arc" and ans.isalpha() else normalise_number(ans)
    if task == "arc":
        letters = re.findall(r"\b([A-E])\b", text)
        return letters[-1] if letters else None
    nums = _NUM.findall(text)
    return normalise_number(nums[-1]) if nums else None


def is_correct(pred: Optional[str], gold: str, task: str = "gsm8k") -> bool:
    if pred is None:
        return False
    if task == "arc":
        return pred.strip().upper() == gold.strip().upper()
    from .data import normalise_number

    try:
        return abs(float(normalise_number(pred)) - float(normalise_number(gold))) < 1e-6
    except ValueError:
        return normalise_number(pred) == normalise_number(gold)
