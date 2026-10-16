# Teaching a small AI with thumbs up and thumbs down

*A plain-words guide to what the Vega feedback work does. No background needed.*

## 1. The AI we worked with

**Vega** is a small AI that makes decisions. You give it a situation and a short list of choices, and it picks one.

> Situation: "I was charged twice. Refund me or I cancel."
> Choices: billing team, technical team, sales team
> Vega: **billing team, 92% sure**

Two things make Vega special:

- **It is small.** It runs on a normal computer. Nothing is sent to a company's servers.
- **It says how sure it is.** Every answer comes with a number like "92% sure".

That second point matters a lot. If an AI says "92% sure", it should be right about 92 times out of 100 when it
says that. An AI whose confidence matches how often it is right is called **calibrated** (think: an honest
weather forecaster whose "70% chance of rain" really means rain 7 days out of 10).

## 2. The problem

Vega comes in two sizes: a small one (0.8B) and a bigger one (4B). On a hard test called **DecisionBench**
(1,075 real-world decision questions):

- the small Vega gets **51 out of 100** right
- the big Vega gets **63 out of 100** right

How do you make it better at *your* job? The usual way is **labels**: someone writes down the correct answer
for hundreds of examples, and the AI studies them. That is slow and costs money, because experts must sit and
answer questions.

But when an AI is in real use, a cheaper signal comes for free: **thumbs up / thumbs down**. The user says "that
was right" or "that was wrong". Nobody tells the AI what the right answer *would* have been.

**Our question:** can Vega learn from only thumbs up and thumbs down, and get as good as if it had the real answers?

## 3. What we did

Think of a student taking practice quizzes. The teacher never shows the answer key. The student only hears
"correct" or "wrong" after each answer. Our method does three things.

**1. Add a small "notebook" to Vega.** We do not change Vega itself. We clip on a tiny extra part (called an
*adapter*). It is tiny, about 400,000 learned numbers next to Vega's 800 million, and it learns while Vega stays the same.

**2. Learn from right and wrong.** Vega answers a question, sometimes trying a less obvious choice, and hears
"right" or "wrong". Answers that got a thumbs up become more likely, and answers that got a thumbs down become less likely. This
kind of learning-by-reward is called **reinforcement learning**.

**3. Keep it on a leash.** Learning from thumbs alone is shaky. The AI can over-react and forget what it already
knew. So we add a rule: "don't drift too far from the original Vega" (called a *KL anchor*). It is like a
student who keeps their old notes while learning new tricks.

**Plus: an honesty fix.** After learning, the "how sure" numbers can be off. We fix them using thumbs again:
if Vega says "80% sure" but gets a thumbs up only 60% of the time, we turn its confidence down until the numbers
match. No answer key is needed for this either.

## 4. What happened

We tested on the same 1,075 questions Vega's makers use, so the comparison is fair.

| | correct out of 100 | how honest its confidence is (lower = more honest) |
|---|---|---|
| small Vega, as released | 51 | 0.10 |
| small Vega + our thumbs-up learning | **61** | **0.05** |
| big Vega, as released | 63.5 | 0.10 |
| big Vega + our thumbs-up learning | **69.5** | **0.045** |

In plain words:

- **The small Vega went from 51 to 61 correct.** That closes most of the gap to the big Vega, which is 5 times
  larger and slower.
- **The big Vega also improved, from 63.5 to 69.5.**
- **Its confidence became about twice as honest** at both sizes.
- **Thumbs alone did about as well as full answer keys.** We also trained the same notebook with the real answers.
  The small Vega did about 1.5 points better with answer keys, and the big Vega about 1 point *worse*. Neither gap
  is big enough to be sure it is real.
  So the cheap signal was enough.
- **It beat Vega's own built-in way of learning** (called TTT) by 8 points on the small model and 6 points on the
  big one. TTT needs answer keys, and on the big model it barely helped at all.
- **It costs almost nothing to run.** Each decision takes less than 1 thousandth of a second longer.

**Someone else checked it.** A different person ran everything again on a different computer. They got +10.4
points where we had reported +10.2. Results that hold up when someone else repeats them are much more
trustworthy.

## 5. Two checks that make the result believable

**"Is Vega's knowledge doing the work, or just the learning trick?"** We gave the same thumbs-up learning to a
simple model that does not have Vega's decision part. It got only 26 out of 100. So the gain comes from
Vega's knowledge *plus* the feedback, not from the trick alone.

**"How much feedback do you need?"** With 5 thumbs per type of question: +3 points. With 60: +11 points. More
feedback, steadily better. It also helped a little (+4 points) on types of questions it never got feedback on.

## 6. How we got here: a detour through poker

We started with a game, not with DecisionBench. We let Vega play a tiny version of poker against copies of itself
and learn from winning and losing. At first it fell apart within a few rounds, always betting no matter its cards.
The "leash" rule fixed that, and that is where the idea came from.

But a fair test showed something humbling. A dumb lookup table with the same leash played poker even better. In
poker, the leash did the work, not Vega's knowledge. So we moved to real-world decisions, where Vega's knowledge
does matter, and the method worked.

## 7. What we have not shown yet

Being honest about the limits is part of the work:

- **The thumbs were simulated.** We made them from the answer key, so they were always correct. Real users are
  messier: they skip giving feedback, click by mistake, or only complain when something is wrong. This is the most
  important thing left to test.
- **One test so far.** Everything above is DecisionBench. Vega's makers also test on 43 public tasks and a
  benchmark called JevBench. Our new script runs those too, but the results are not in yet.
- **Trying alternatives has a cost.** To learn, Vega sometimes picks a less likely answer on purpose. In real use,
  some of those will be wrong. We have not measured how much that costs.

## 8. Why this could matter

Many businesses want a small AI they can run privately, which improves from everyday use, without paying experts
to write answer keys. This work shows a way to do that: let users react, learn from the reactions, keep the model
on a leash, and keep its confidence honest. It ran on a laptop, the extra part is tiny, and it held up when
someone else repeated it.

---

### Words used here

- **Model / AI**: a program that learned patterns from lots of text.
- **0.8B / 4B**: model size, 0.8 billion or 4 billion learned numbers. Bigger is usually smarter and slower.
- **Benchmark**: a fixed test set used to compare AIs fairly.
- **Labels / answer key**: the correct answers, written by people.
- **Feedback / reward**: only "right" or "wrong", with no correct answer given.
- **Reinforcement learning**: learning by trying things and getting rewards.
- **Adapter**: a small add-on that learns, while the main model stays unchanged.
- **KL anchor ("leash")**: a rule that stops the learner from drifting far from the original model.
- **Calibration**: when "80% sure" really means right 80% of the time.
- **Seed**: one run of the experiment with different randomness. We used 3 to make sure a result was not luck.
