# Vendored verbatim from NandhaKishorM/vegaml@00d3d99, notebooks/vega_benchmark_2xT4.ipynb (Apache-2.0).
# Do not edit: this is VegaML's own benchmark harness, used so every number is loaded and scored as theirs is.
"""The public evaluation tasks, each expressed as a typed decision Vega and Jev can both be asked.

Three shapes cover everything here:

* **fixed-option** tasks (intent, topic, sentiment, safety): one question per task, the options are
  the label set, the row supplies the state. 77-way and 151-way tasks are in on purpose, because a
  wide label space with few examples per class is where a fitted head stops helping.
* **per-row-option** tasks (multiple choice): the options differ row by row, so the question is
  built per row and the option **ids** are kept stable (A, B, C ...) so one head can span them.
* **ordinal** tasks: answered as `score`, which is the only type that carries an order, so an
  ordinal mistake of one level is not scored the same as a mistake of three.

Each entry records its upstream licence. Several are non-commercial or research-only, which is why
this harness evaluates on them and never writes their text into its output.
"""
import hashlib
import json
import random

CHOICE_IDS = [chr(ord("A") + i) for i in range(26)]


# ----------------------------------------------------------------------------- helpers
def _names(ds, col):
    f = ds.features.get(col)
    return list(getattr(f, "names", []) or [])


def _text(row, cols):
    return {c: row[c] for c in cols if row.get(c) is not None}


def fixed(key, hf, config, label_col, state_cols, instructions, licence,
          qtype="choice", criteria=None, labels=None, eval_split="test", train_split="train",
          gold_map=None, state_fn=None, group=None, note=None):
    """A task whose option set is the same for every row."""
    return {"key": key, "hf": hf, "config": config, "kind": "fixed", "qtype": qtype,
            "label_col": label_col, "state_cols": state_cols, "instructions": instructions,
            "licence": licence, "criteria": criteria, "labels": labels,
            "eval_split": eval_split, "train_split": train_split, "gold_map": gold_map,
            "state_fn": state_fn, "group": group or "classification", "note": note}


def perrow(key, hf, config, instructions, licence, state_fn, options_fn, gold_fn,
           eval_split="test", train_split="train", group="multiple_choice", note=None,
           qtype="choice"):
    """A multiple-choice task: options come from the row, ids stay A, B, C ... so a head can span them."""
    return {"key": key, "hf": hf, "config": config, "kind": "perrow", "qtype": qtype,
            "instructions": instructions, "licence": licence, "state_fn": state_fn,
            "options_fn": options_fn, "gold_fn": gold_fn, "eval_split": eval_split,
            "train_split": train_split, "group": group, "note": note}


# ----------------------------------------------------------------------------- the registry
YESNO = {"false": "no", "true": "yes"}

TASKS = [
    # ------------------------------------------------- intent and topic, wide label spaces
    fixed("banking77", "mteb/banking77", None, "label_text", ["text"],
          "Which banking intent does this customer utterance express?", "cc-by-4.0",
          group="intent", note="77 intents: the wide-label-space case. The mteb mirror is used "
                               "because PolyAI/banking77 still ships a loading script"),
    fixed("clinc150_oos", "clinc/clinc_oos", "plus", "intent", ["text"],
          "Which intent does this utterance express, or is it out of scope?", "cc-by-3.0",
          group="intent", note="150 intents plus an out-of-scope class"),
    fixed("massive_intent_en", "mteb/amazon_massive_intent", "en", "label_text", ["text"],
          "Which assistant intent does this utterance express?", "apache-2.0",
          group="intent", note="60 intents, spoken-assistant phrasing"),
    fixed("ledgar", "coastalcph/lex_glue", "ledgar", "label", ["text"],
          "Which contract provision type is this clause?", "cc-by-4.0",
          group="legal", note="100 provision types"),
    fixed("ag_news", "fancyzhx/ag_news", None, "label", ["text"],
          "Which section of the newspaper does this story belong to?", "unknown-see-card",
          group="topic"),
    fixed("dbpedia14", "fancyzhx/dbpedia_14", "dbpedia_14", "label", ["title", "content"],
          "Which ontology class does this article describe?", "cc-by-sa-3.0", group="topic"),
    fixed("trec_coarse", "CogComp/trec", None, "coarse_label", ["text"],
          "What kind of answer does this question ask for?", "unknown-see-card", group="topic"),
    fixed("emotion", "dair-ai/emotion", "split", "label", ["text"],
          "Which emotion does the writer express?", "other-see-card", group="affect"),

    # ------------------------------------------------- entailment, verification, policy
    fixed("anli_r3", "facebook/anli", "plain_text", "label", ["premise", "hypothesis"],
          "Does the premise entail the hypothesis, contradict it, or neither?", "cc-by-nc-4.0",
          eval_split="test_r3", train_split="train_r3", group="entailment",
          note="adversarial NLI, round 3"),
    fixed("glue_rte", "nyu-mll/glue", "rte", "label", ["sentence1", "sentence2"],
          "Does the first sentence entail the second?", "other-see-card",
          qtype="boolean", criteria={"false": "it does not entail it",
                                     "true": "it entails it"},
          labels=["true", "false"], eval_split="validation", group="entailment",
          gold_map={"entailment": "true", "not_entailment": "false"}),
    fixed("superglue_cb", "aps/super_glue", "cb", "label", ["premise", "hypothesis"],
          "Does the premise entail the hypothesis, contradict it, or neither?", "other-see-card",
          eval_split="validation", group="entailment"),
    fixed("vitaminc", "tals/vitaminc", None, "label", ["claim", "evidence"],
          "Does the evidence support the claim, refute it, or say too little to tell?",
          "cc-by-sa-3.0", group="verification",
          labels=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]),
    fixed("climate_fever", "tdiggelm/climate_fever", None, "claim_label", ["claim"],
          "Does the evidence support the claim, refute it, say too little, or disagree with itself?",
          "unknown-see-card", eval_split="test", train_split="test", group="verification",
          state_fn=lambda row, ctx: {
              "claim": row["claim"],
              "evidence": [e["evidence"] for e in (row.get("evidences") or [])][:5]},
          note="one split upstream, so eval and fit rows are disjoint halves"),
    fixed("boolq", "google/boolq", None, "answer", ["passage", "question"],
          "Is the answer to the question yes, according to the passage?", "cc-by-sa-3.0",
          qtype="boolean", criteria=YESNO, labels=["true", "false"], eval_split="validation",
          group="reading", gold_map={True: "true", False: "false", "True": "true", "False": "false"}),

    # ------------------------------------------------- safety, abuse, injection
    fixed("toxic_chat_toxicity", "lmsys/toxic-chat", "toxicchat0124", "toxicity", ["user_input"],
          "Is this user message toxic?", "cc-by-nc-4.0", qtype="boolean",
          criteria={"false": "ordinary, even if blunt or rude",
                    "true": "hateful, harassing, sexual, violent or otherwise toxic"},
          labels=["true", "false"], group="safety",
          gold_map={1: "true", 0: "false", "1": "true", "0": "false"}),
    fixed("toxic_chat_jailbreak", "lmsys/toxic-chat", "toxicchat0124", "jailbreaking",
          ["user_input"], "Is this user message trying to jailbreak the assistant?",
          "cc-by-nc-4.0", qtype="boolean",
          criteria={"false": "an ordinary request, however unusual",
                    "true": "it tries to override the assistant's rules or role"},
          labels=["true", "false"], group="safety",
          gold_map={1: "true", 0: "false", "1": "true", "0": "false"}),
    fixed("aegis_prompt_safety", "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", None,
          "prompt_label", ["prompt"], "Is this prompt unsafe to answer as written?", "cc-by-4.0",
          qtype="boolean", criteria={"false": "safe to answer", "true": "unsafe to answer"},
          labels=["true", "false"], group="safety",
          gold_map={"unsafe": "true", "safe": "false"}),
    fixed("prompt_injections", "deepset/prompt-injections", None, "label", ["text"],
          "Is this input a prompt injection attempt?", "apache-2.0", qtype="boolean",
          criteria={"false": "an ordinary request or statement",
                    "true": "it tries to override the instructions the system was given"},
          labels=["true", "false"], group="safety",
          gold_map={1: "true", 0: "false", "1": "true", "0": "false"}),
    fixed("jailbreak_cls", "jackhhao/jailbreak-classification", None, "type", ["prompt"],
          "Is this prompt a jailbreak attempt?", "apache-2.0", qtype="boolean",
          criteria={"false": "a benign prompt", "true": "a jailbreak attempt"},
          labels=["true", "false"], group="safety",
          gold_map={"jailbreak": "true", "benign": "false"}),
    fixed("enron_spam", "SetFit/enron_spam", None, "label_text", ["subject", "message"],
          "Is this email spam?", "unknown-see-card", qtype="boolean",
          criteria={"false": "wanted mail", "true": "unsolicited bulk mail"},
          labels=["true", "false"], group="safety",
          gold_map={"spam": "true", "ham": "false"}),
    fixed("unfair_tos", "coastalcph/lex_glue", "unfair_tos", "labels", ["text"],
          "Does this terms-of-service clause contain a potentially unfair term?", "cc-by-4.0",
          qtype="boolean", criteria={"false": "no unfair term", "true": "at least one unfair term"},
          labels=["true", "false"], group="legal",
          state_fn=lambda row, ctx: {"clause": row["text"]},
          gold_map="nonempty_list"),
    fixed("tweet_hate", "cardiffnlp/tweet_eval", "hate", "label", ["text"],
          "Is this tweet hate speech?", "unknown-see-card", qtype="boolean",
          criteria={"false": "not hate speech", "true": "hate speech"},
          labels=["true", "false"], group="affect",
          gold_map={1: "true", 0: "false"}),
    fixed("tweet_offensive", "cardiffnlp/tweet_eval", "offensive", "label", ["text"],
          "Is this tweet offensive?", "unknown-see-card", qtype="boolean",
          criteria={"false": "not offensive", "true": "offensive"},
          labels=["true", "false"], group="affect", gold_map={1: "true", 0: "false"}),
    fixed("tweet_irony", "cardiffnlp/tweet_eval", "irony", "label", ["text"],
          "Is this tweet ironic?", "unknown-see-card", qtype="boolean",
          criteria={"false": "said straight", "true": "ironic"},
          labels=["true", "false"], group="affect", gold_map={1: "true", 0: "false"}),
    fixed("tweet_stance_abortion", "cardiffnlp/tweet_eval", "stance_abortion", "label", ["text"],
          "What stance does this tweet take on the topic?", "unknown-see-card", group="affect"),

    # ------------------------------------------------- ordinal, answered as a score
    fixed("tweet_sentiment", "cardiffnlp/tweet_eval", "sentiment", "label", ["text"],
          "How positive is this tweet?", "unknown-see-card", qtype="score",
          criteria=["negative", "neutral", "positive"], group="ordinal",
          gold_map={0: "0", 1: "1", 2: "2"}),
    fixed("yelp_stars", "Yelp/yelp_review_full", "yelp_review_full", "label", ["text"],
          "How many stars does this review give?", "yelp-tos-research-only", qtype="score",
          criteria=["one star, very negative", "two stars", "three stars", "four stars",
                    "five stars, very positive"], group="ordinal",
          note="five ordered levels: the widest ordinal scale here",
          gold_map={i: str(i) for i in range(5)}),

    # ------------------------------------------------- multiple choice, per-row options
    perrow("arc_challenge", "allenai/ai2_arc", "ARC-Challenge",
           "Which option answers the question correctly?", "cc-by-sa-4.0",
           lambda r, c: {"question": r["question"]},
           lambda r, c: list(r["choices"]["text"]),
           lambda r, c: _letter_gold(r["answerKey"], list(r["choices"]["label"])),
           group="science"),
    perrow("arc_easy", "allenai/ai2_arc", "ARC-Easy",
           "Which option answers the question correctly?", "cc-by-sa-4.0",
           lambda r, c: {"question": r["question"]},
           lambda r, c: list(r["choices"]["text"]),
           lambda r, c: _letter_gold(r["answerKey"], list(r["choices"]["label"])),
           group="science"),
    perrow("openbookqa", "allenai/openbookqa", "main",
           "Which option answers the question correctly?", "unknown-see-card",
           lambda r, c: {"question": r["question_stem"]},
           lambda r, c: list(r["choices"]["text"]),
           lambda r, c: _letter_gold(r["answerKey"], list(r["choices"]["label"])),
           group="science"),
    perrow("sciq", "allenai/sciq", None,
           "Which option answers the science question correctly?", "cc-by-nc-3.0",
           lambda r, c: {"question": r["question"]},
           lambda r, c: _shuffled(c, [r["correct_answer"], r["distractor1"], r["distractor2"],
                                      r["distractor3"]])[0],
           lambda r, c: _shuffled(c, [r["correct_answer"], r["distractor1"], r["distractor2"],
                                      r["distractor3"]])[1],
           group="science"),
    perrow("commonsenseqa", "tau/commonsense_qa", None,
           "Which option answers the question correctly?", "mit",
           lambda r, c: {"question": r["question"]},
           lambda r, c: list(r["choices"]["text"]),
           lambda r, c: _letter_gold(r["answerKey"], list(r["choices"]["label"])),
           eval_split="validation", group="commonsense"),
    perrow("hellaswag", "Rowan/hellaswag", None,
           "Which ending continues the passage most plausibly?", "mit",
           lambda r, c: {"context": r["ctx"], "activity": r["activity_label"]},
           lambda r, c: list(r["endings"]),
           lambda r, c: CHOICE_IDS[int(r["label"])] if str(r["label"]) != "" else None,
           eval_split="validation", group="commonsense",
           note="long options: the 32-token option cap bites here, see the budget audit"),
    perrow("winogrande", "allenai/winogrande", "winogrande_xl",
           "Which name does the blank in the sentence refer to?", "unknown-see-card",
           lambda r, c: {"sentence": r["sentence"]},
           lambda r, c: [r["option1"], r["option2"]],
           lambda r, c: CHOICE_IDS[int(r["answer"]) - 1],
           eval_split="validation", group="commonsense"),
    perrow("mmlu", "cais/mmlu", "all",
           "Which option answers the question correctly?", "mit",
           lambda r, c: {"subject": r["subject"], "question": r["question"]},
           lambda r, c: list(r["choices"]),
           lambda r, c: CHOICE_IDS[int(r["answer"])],
           train_split="validation", group="knowledge"),
    perrow("mmlu_pro", "TIGER-Lab/MMLU-Pro", None,
           "Which option answers the question correctly?", "mit",
           lambda r, c: {"category": r["category"], "question": r["question"]},
           lambda r, c: list(r["options"]),
           lambda r, c: CHOICE_IDS[int(r["answer_index"])],
           train_split="validation", group="knowledge",
           note="up to ten options per row"),
    perrow("medmcqa", "openlifescienceai/medmcqa", None,
           "Which option answers the medical question correctly?", "apache-2.0",
           lambda r, c: {"question": r["question"]},
           lambda r, c: [r["opa"], r["opb"], r["opc"], r["opd"]],
           lambda r, c: CHOICE_IDS[int(r["cop"])],
           eval_split="validation", group="medical"),
    perrow("medqa_usmle", "GBaker/MedQA-USMLE-4-options", None,
           "Which option answers the clinical question correctly?", "cc-by-4.0",
           lambda r, c: {"vignette": r["question"]},
           lambda r, c: [r["options"][k] for k in sorted(r["options"])],
           lambda r, c: CHOICE_IDS[sorted(r["options"]).index(r["answer_idx"])],
           group="medical", note="long clinical vignettes"),
    perrow("pubmedqa", "qiaojin/PubMedQA", "pqa_labeled",
           "Does the abstract answer the research question yes, no, or maybe?", "mit",
           lambda r, c: {"question": r["question"],
                         "abstract": " ".join(r["context"]["contexts"])[:4000]},
           lambda r, c: ["yes", "no", "maybe"],
           lambda r, c: {"yes": "A", "no": "B", "maybe": "C"}.get(str(r["final_decision"]).lower()),
           eval_split="train", train_split="train", group="medical",
           note="one split upstream, so eval and fit rows are disjoint halves"),
    perrow("casehold", "coastalcph/lex_glue", "case_hold",
           "Which holding statement fits the citation in this excerpt?", "cc-by-4.0",
           lambda r, c: {"excerpt": r["context"]},
           lambda r, c: list(r["endings"]),
           lambda r, c: CHOICE_IDS[int(r["label"])],
           group="legal", note="five long holdings: the clearest test of the option cap"),
    perrow("race_middle", "ehovy/race", "middle",
           "Which option answers the question about the passage?", "other-see-card",
           lambda r, c: {"passage": r["article"], "question": r["question"]},
           lambda r, c: list(r["options"]),
           lambda r, c: str(r["answer"]).strip().upper(),
           group="reading", note="long passages"),
    perrow("truthfulqa_mc1", "truthfulqa/truthful_qa", "multiple_choice",
           "Which answer is true?", "apache-2.0",
           lambda r, c: {"question": r["question"]},
           lambda r, c: list(r["mc1_targets"]["choices"])[:12],
           lambda r, c: CHOICE_IDS[list(r["mc1_targets"]["labels"]).index(1)]
           if 1 in list(r["mc1_targets"]["labels"])[:12] else None,
           eval_split="validation", train_split="validation", group="knowledge",
           note="one split upstream, so eval and fit rows are disjoint halves"),
    perrow("cruxeval_output", "cruxeval-org/cruxeval", None,
           "Which value does the function return for the given input?", "mit",
           lambda r, c: {"code": r["code"], "input": r["input"]},
           lambda r, c: _crux_options(r, c)[0],
           lambda r, c: _crux_options(r, c)[1],
           eval_split="test", train_split="test", group="code",
           note="distractors drawn from other rows' outputs with a fixed seed; "
                "eval and fit rows are disjoint halves"),
]


# ----------------------------------------------------------------------------- per-row helpers
def _letter_gold(answer_key, labels):
    """ARC-style rows label their own options (A-D, or 1-4), so map the key through that list."""
    key = str(answer_key).strip()
    labs = [str(x).strip() for x in labels]
    if key in labs:
        return CHOICE_IDS[labs.index(key)]
    return None


def stable_hash(value):
    """A hash that is the same in every process. `hash()` is salted per run, so seeding a shuffle
    from it would quietly make the sampled rows and the option order differ between runs."""
    blob = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode()
    return int(hashlib.md5(blob).hexdigest()[:8], 16)


def _rowrng(ctx, row):
    """A per-row generator seeded by the run seed and the row content, so a shuffle is reproducible
    and does not depend on the order rows arrive in."""
    return random.Random(stable_hash([ctx.get("seed", 0),
                                      json.dumps(row, sort_keys=True, default=str)[:400]]))


def _shuffled(ctx, options):
    """Options in a row-stable random order, plus the id the first option ended up at.

    SciQ lists the correct answer first. Asking in that order measures position, not science.
    """
    rng = _rowrng(ctx, {"o": options})
    order = list(range(len(options)))
    rng.shuffle(order)
    return [options[i] for i in order], CHOICE_IDS[order.index(0)]


def _crux_options(row, ctx):
    """CRUXEval is a generation task upstream. Here it becomes four-way multiple choice: the real
    return value plus three return values from other rows, which keeps it answerable by a model that
    cannot emit text. The distractor pool is passed in by the loader."""
    pool = [o for o in ctx.get("crux_pool", []) if o != row["output"]]
    rng = _rowrng(ctx, {"i": row.get("id"), "c": row.get("input")})
    picks = rng.sample(pool, 3) if len(pool) >= 3 else pool
    opts, gold = _shuffled({"seed": ctx.get("seed", 0)}, [row["output"]] + list(picks))
    return opts, gold


BY_KEY = {t["key"]: t for t in TASKS}
GROUPS = sorted({t["group"] for t in TASKS})
