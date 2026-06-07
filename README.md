# Popping the Bubble: Surfacing Diverse Perspectives on Social Media

This repo contains a prototype system that, given a social media post expressing an opinion about a topic, returns a small set of representative statements covering *other* arguments people are making about that topic. The goal is to help surface arguments people wouldn't see by default on social media. Much of what we see on social media tends to reinforce what we already believe (either because its presented to us via algorithms that favour content we're likely to agree with, or just because we tend to click more on posts that agree with our existing beliefs).

This repo contains a pipeline (claim extraction → topic clustering → sub-topic discovery → polarity assignment → slate generation), a small FastAPI backend, and a (vibe coded) React frontend. It runs entirely on a laptop, or against any OpenAI-compatible LLM endpoint.

I wrote up the motivation and original design plan in [Popping the Bubble](https://www.j-roy.com/blog/03_chatJTP/sm_polarization_project) on my blog. The system has shifted quite a bit since then; the most up-to-date description is in this README.

## Try it

> **Demo site:** *coming soon* — I'll host a small public instance so you can paste a post and see what comes back.
>
> **Sample output / screenshots:** *also coming soon. :)*

If you'd rather run it locally, jump to [Run it locally](#run-it-locally). You'll need to first do the batch run -- this can be done on a laptop (using a llamafile for the LLM), but it will just take some time; It can also be done with any openAI compatible LLM ednpoint (I used together AI for roughly 30 cents to do the full batch of 1500 posts). I plan to host a query-able version of the graph in the near future so that it can be used locally without needing to fully re-run the full batch. 

## Why

Social media tends to give rise to echo chambers — whether steered by algorithms or by self-selection — where the content a user sees largely aligns with their existing values and beliefs, and tends to amplify more extreme versions of them. The evidence on how widespread this is is genuinely mixed, but for the subset of users who do end up in reinforced bubbles, the effects can be meaningful: more polarization, less critical examination of one's own views, less exposure to people who disagree in good faith.

The ability to independently shape our knowledge, beliefs, and understanding is sometimes called *epistemic agency*, and it's essential to meaningful participation in democratic societies. Our vote — with ballots, our wallets, or our actions — means very little if the opinions behind it were shaped *for* us rather than *by* us.

Most ranking algorithms filter content so that feeds are more aligned with existing views. This project tries to do the opposite: bring in content that *differs* from or *opposes* the user's stance, in the hope it nudges them to question and refine it. Whether they keep their original stance, hold it less strongly, or change their mind is up to them — the point is that the choice is theirs.

## What it does

Given a corpus of posts on contested subjects (currently public Mastodon hashtag timelines), the system builds a browsable hierarchy:

```
Topic  ──►  Sub-topic  ──►  Agree / Disagree slate
                               (3 representative statements per side)
```

A *Topic* is a subject like "Climate Change" or "Cats vs Dogs". A *Sub-topic* is one axis of disagreement *inside* that topic — e.g. "Veganism" splits into ethical, health, and environmental sub-topics, each with its own claim. Each sub-topic carries a declarative `polarity_target` ("Veganism is the ethical choice"), and each side is summarised by a *slate* of 3 statements drawn from real claims, rather than collapsed into a single hedged sentence.

At query time, a user pastes in a post; the system embeds it, finds the closest topic(s) by cosine similarity to a precomputed centroid, and shows the slates for the topic the user picks. The query path makes **zero LLM calls** — all the heavy lifting is done in batch ahead of time.

> *(System diagram: high-level flow from post → topic match → sub-topic → agree/disagree slates. To be added — the old one in [the blog post](https://www.j-roy.com/blog/03_chatJTP/sm_polarization_project) is the right shape but predates the sub-topic layer.)*

## How it works

The pipeline is split into five batch stages and a real-time query path. There's a much fuller walkthrough in [`docu/system-summary.md`](docu/system-summary.md); the summary below is the shape of it.

```
┌─────────────────────────── BATCH (LLM-heavy, run intermittently) ───────────────────────┐
│                                                                                         │
│  posts ──► claim extraction ──► topic clustering ──► sub-topic discovery ──►            │
│             (LLM, JSON schema)   (BERTopic on         (UMAP+HDBSCAN +                   │
│                                   claim embeddings)    LLM framing on                   │
│                                                        sample of real claims)           │
│                                                                                         │
│             ──► polarity assignment ──► slate generation                                │
│                  (LLM, per-claim          (GEN/DISC loop, à la                          │
│                   agree/disagree)          Generative Social Choice)                    │
│                                                                                         │
│  writes:  data/pipeline.db  +  data/bertopic_model                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────── QUERY (LLM-free, 24/7) ──────────────────────────────────────┐
│                                                                                         │
│  user post ──► embedding ──► top-k centroid match ──► user picks topic ──►              │
│                                                                                         │
│             ──► render every sub-topic's agree/disagree slate (DB read only)            │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Claim extraction

For each post, a small LLM returns a list of opinion *claims* — short, self-contained propositions — plus a topic-naming sentence. Posts with no opinion claims (greetings, jokes, news links, etc.) drop out cleanly here.

### 2. Topic clustering

[BERTopic](https://maartengr.github.io/BERTopic/) over the extracted claims: sentence embeddings → UMAP → HDBSCAN → c-TF-IDF labels, with the LLM rewriting cluster keywords into clean 2-5 word topic titles. Clustering happens at the *claim* level, not the post level, so a post that makes two unrelated arguments shows up under both relevant topics rather than being forced into one bucket.

### 3. Sub-topic discovery

Inside each topic, claims are clustered a second time using a preference-aware embedding model ([`cartgr/embeddings-for-preferences-st5-xl`](https://huggingface.co/cartgr/embeddings-for-preferences-st5-xl)) that groups by *opinion expressed* rather than just by surface vocabulary. The LLM then reads a sample of central claims from each sub-cluster and either (a) writes a declarative `polarity_target` like *"Remote work is sustainable"*, or (b) marks the cluster *descriptive* if no single proposition fits — in which case it skips polarity assignment entirely.

This was the biggest structural change from the original plan: forcing one agree/disagree axis per topic was the root cause of slates that all sounded like *"yes, but it's complex"*. Full rationale in [`docu/why-subtopics.md`](docu/why-subtopics.md).

### 4. Polarity assignment

For each claim under a non-descriptive sub-topic, the LLM classifies it as `agree` or `disagree` with the sub-topic's `polarity_target`. One bounded binary classification per claim — small enough to be reliable on a small model.

### 5. Slate generation (GEN/DISC)

For each `(sub_topic, polarity)` bucket, a Generative-Social-Choice-style [GEN/DISC](https://arxiv.org/abs/2309.01291) loop produces a slate of 3 representative statements. GEN proposes a statement that should represent some likely-to-agree subgroup; DISC scores how well the statement covers remaining claims (we approximate the paper's survey-based DISC with preference-aware embeddings); the best-covered claims are removed; repeat. Removing covered claims is what makes the slate *pluralistic* — you get several coherent sub-views rather than one averaged-out sentence.

### Real-time query path

Two endpoints, no LLM calls:

| Endpoint | What it does |
|---|---|
| `POST /match-post-topics?k=5` | Embed the post; return the top-k topics by centroid similarity. |
| `GET /topic-response/{id}` | Return the precomputed agree/disagree slates for every sub-topic of the chosen topic. |

The two-step shape exists so the user can correct the framing when the top match isn't quite right (a common case for ambiguous posts).

## Why this didn't need a frontier LLM

A central design constraint was that I wanted the system runnable on a laptop, or at most against a small hosted 8B model — not because frontier models can't do this end-to-end (they probably could, albeit for a limited subset of posts/topics and opinions), but because *running an LLM on every social media post is wasteful when the problem decomposes nicely*. There are billions of posts a day; a real deployment of something like this can't be sending each one through a frontier model.

The trick is to break the problem into pieces small enough that a small model handles each piece well, and to use rigorous, deterministic methods for the structural work between LLM calls. Concretely:

- **Each LLM call is narrowly scoped.** Some examples are: "Extract claims from one post. Rewrite this keyword string into a topic title. Read 20 claims and pick the dominant proposition. Decide whether *this* claim agrees or disagrees with *this* statement. None of these is a "do the whole task" prompt"; each of these is a small, well-bounded transform -- which is doable with a small (non-frontier) model. 
- **All structured outputs go through JSON-schema constrained sampling.** I had never heard of grammar for LLMs prior to this project but its pretty cool! In the end I used a json format, which is similar conceptually (modifies the probabilities of output tokens). This means we don't need regex parsing of free-form text; the provider enforces the shape.
- **The deterministic parts stay deterministic.** Embeddings, UMAP, HDBSCAN, centroid similarity, the relational schema, the GSC removal rule — all of this is mechanical, reproducible, and doesn't depend on what the model felt like saying that day.
- **The structure of the pipeline (Topic → Sub-topic → agree/disagree) *is* the prior.** It tells the LLM, by construction, what kind of decisions to make and at what granularity. This means we (humans) stay in the loop and can understand + debug each step! I think its so important that we build systems that we understand, rather than delegating decisions and 'critical thinking' to LLMs. 
- **Things drop out cleanly when there's nothing to say.** Posts with no opinion claims drop out at stage 1. Sub-topics with no coherent axis of disagreement set `polarity_target = NULL` and skip polarity assignment. Buckets with zero claims produce no slate. This cleans up the 'messy' short text social media posts. The posts used to 'train' this system (ie. those loaded into the argument graph) were not pre-processed whatsoever beyond loading them from mastodon and passing them to the pipeline. 

The result is that the system handles the kinds of inputs LLMs are genuinely good at (short, contextual judgments) and lets BERTopic, UMAP, HDBSCAN, embeddings, and SQL handle everything else. Even on noisy short-form Mastodon posts — which most pipelines mishandle — this is enough to produce a real, navigable map of the debate.

The default local model is Llama-3.2-3B via [llamafile](https://github.com/Mozilla-Ocho/llamafile); the hosted path runs against any OpenAI-compatible 8B endpoint (Together AI, Fireworks, DeepInfra, …). A full 1500-post batch on hosted 8B is on the order of single-digit cents and finishes in ~10 minutes.

## Status & scope

This is a **prototype**. It is intentionally scoped to one platform (Mastodon, via the public hashtag timeline API) and a few thousand posts (bc I'm just a girl and cannot unfortunately afford the compute for more on my own :'. The point isn't to ship something production-ready — it's to show that the shape of the system works on real, messy social-media data, and that the expensive pieces (LLM batch) can be cleanly separated from the cheap pieces (query-time inference) so a real deployment is plausible. I hope that this project can start a conversation about using systems like this -- both on social media and in other settings where it is useful to represent diverse opinions about different topics. 

I don't personally have experience dealing with datasets on the scale of social media, but I did my best to design the architecture to be scalable (ish) to bigger datasets:

- **Batch LLM work + LLM-free query path.** The slates are precomputed and cached; serving them is a SQLite read. A real platform integration would re-batch periodically, not call an LLM per post.
- **No GPU required at serve time.** A tiny CPU PaaS box is enough.
- **One pipeline runs against any provider.** Local llamafile for offline dev, hosted 8B for batch runs, swap via an env var.

The biggest limitation remaining is the inability to 'add new topics' -- so if we get new data from social media (eg. a big new batch of posts). we need to re-cluster _everything_ rather than adding on to the data that was already clustered, had statements, etc. This is very inneficient and potentially prohibitive, particularly since there are so many new social media posts and topics being discussed all the time. This is the next thing on my radar, so I hope to have a cool workaround to this soon! 

## Run it locally

Requires Docker. Pick one of the two LLM backends.

**Hosted (recommended — much faster, the 1500 post dataset cost me around 30 cents per batch, which IMO was worth it since it saved many hours and was doable with the free credits together AI gave me!):**

```bash
# .env, next to docker-compose.yml
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.together.xyz/v1
LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct-Lite
LLM_API_KEY=tgp-...
LLM_CONCURRENCY=8
```

```bash
docker compose up -d
docker compose exec pipeline python main.py     # build data/pipeline.db
open http://localhost:8000
```

**Fully local with llamafile (free, slow on CPU — multi-hour batches):**

```bash
docker compose --profile local up -d
docker compose exec pipeline python main.py
```

Either way, the frontend is served by the API container at `http://localhost:8000/`. There are two pages:

- **Generate** — paste a post, pick from the closest topics, see the agree/disagree slates for the topic you choose.
- **Graph** — browse the whole `Topic → Sub-topic → agree/disagree → claim` hierarchy as an expandable radial graph.

If you have thoughts or feedback on any of this, I'd love to hear them — contact info is in the footer of [my site](https://www.j-roy.com/). You can also make an issue on this repo and I will do my best to respond promptly! 
