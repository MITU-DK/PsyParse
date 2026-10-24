import config
from evaluation.run_eval import make_judge, judge_transcript

# the judge has to clearly prefer the strong transcript, otherwise the A/B deltas mean nothing
weak = (
    "Patient: I keep thinking everyone on the high street is staring at me, so I keep my head down.\n"
    "Therapist: That sounds hard. How was your day otherwise?\n"
    "Patient: Not great, I just rushed home.\n"
    "Therapist: I see. Try to stay positive and things will get better.\n"
    "Patient: I guess.\n"
    "Therapist: Is there anything else you want to talk about?"
)
strong = (
    "Patient: I keep thinking everyone on the high street is staring at me, so I keep my head down.\n"
    "Therapist: Walking down the high street feeling watched sounds exhausting, and keeping your head down makes sense as a way to protect yourself. "
    "When you notice that thought, 'everyone is staring at me', how strongly do you believe it, from 0 to 100?\n"
    "Patient: About 90. I feel like they're judging how I look.\n"
    "Therapist: So at 90 percent sure, no wonder you rush your shopping. Let's look at that thought together like a detective: "
    "what have you actually seen people doing on the street that supports it, and what doesn't fit it?\n"
    "Patient: Honestly, most people are looking at their phones or the shops. I don't really know if anyone looks at me.\n"
    "Therapist: That's an important observation. If you held the thought 'most people are busy with their own errands', how much would you believe it now? "
    "Maybe this week we could try a small experiment: lift your head for just one shop window and notice what people are really doing."
)

judge = make_judge()
print("judge model:", config.JUDGE_MODEL)
w = judge_transcript(judge, weak)
s = judge_transcript(judge, strong)
print("weak  :", w)
print("strong:", s)
if w and s:
    gap = sum(s.values()) / len(s) - sum(w.values()) / len(w)
    print(f"avg gap (strong - weak) = {gap:+.1f}  ->", "OK" if gap > 15 else "judge is not discriminating well, try a bigger JUDGE_MODEL")
