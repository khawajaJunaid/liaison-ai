# liAIson

*A lease and issue agent for property owners. It reads the paperwork and photos, shows its working,
and leaves every decision to a person.* It sits between the documents and the owner, and links the two
things that are usually kept apart: a unit's lease and the problems reported in it.

A small full-stack service for a property owner (the sample data is Marina Crest Holdings, Doha):

- **Part A, lease record.** Upload a lease PDF. The agent reads it into a structured record where
  every value points back to the page, box and quote it came from, flags what a human should check,
  validates it against the owner's ruleset (PASS / FAIL / NOT_DETERMINABLE with reasons), matches it
  to a unit and, once a person accepts it, marks that unit occupied.
- **Part B, issue reporting.** Upload photos of a problem. The agent assesses condition, lists the
  equipment it sees, and drafts a work order.
- **Together.** Open a unit and see its lease and every issue raised against it on one screen. A person
  can accept, reject or override each extracted field, each flag and each work order. Every decision
  lands in an audit trail.

## Run it

Python 3.12+. No API key is needed for the default setup.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m scripts.make_samples          # writes samples/ (leases + placeholder photos)
uvicorn app.main:app --reload           # http://127.0.0.1:8000
pytest                                  # 145 tests, offline, about 3 seconds
```

Try it with the samples:

| Sample | What to expect |
|---|---|
| `samples/lease_good.pdf` | All 7 rules pass, unit `MC-B-1204` matched. Accept it and the unit turns occupied. |
| `samples/lease_defective.pdf` | Every rule fails: deposit under one month, 48-month term whose dates span 36, vague escalation, unsigned tenant, annual rent that does not reconcile, conflicting start dates, and the unit is already occupied. Accepting is blocked until you correct the unit and decide the high-severity flags. |
| `samples/lease_scanned.pdf` | No text layer. Without OCR it is flagged as unreadable and rules report NOT_DETERMINABLE rather than failing. |
| `samples/photos/*.jpg` | Placeholders. The stub vision model reads the **file name** (`ac_unit_water_leak.jpg`) and your note. |

### API keys and optional engines

| Setting | What it enables | Needs |
|---|---|---|
| default | Everything, with a deterministic stub for photo assessment | nothing |
| `LEASE_AGENT_VISION=anthropic` | Real photo assessment with Anthropic | `pip install anthropic`, `ANTHROPIC_API_KEY`; optional `VISION_MODEL` (default `claude-sonnet-5-5`) |
| `LEASE_AGENT_VISION=openai` | Real photo assessment with OpenAI | `pip install openai`, `OPENAI_API_KEY`; optional `VISION_MODEL` (default `gpt-4o`, change it to whichever vision model your key can use) |
| `LEASE_AGENT_VISION=openai` plus `OPENAI_BASE_URL` | A local or other OpenAI-compatible server (Ollama, vLLM, LM Studio) | no real key; set `VISION_MODEL` to a vision-capable model that server hosts |
| `LEASE_AGENT_OCR=paddle` | OCR for scanned leases (PP-OCR, line-level boxes, CPU). **Verified end to end** | `pip install -r requirements-ocr.txt`; downloads about 20 MB of models on first use; runs locally, no key. `LEASE_AGENT_OCR_LANG=ar` for Arabic (untested) |
| `LEASE_AGENT_OCR=paddle-vl` | PaddleOCR-VL: higher benchmark score, paragraph-level boxes. **Not run here** | needs PaddlePaddle 3.2 or newer (not published for Intel macOS); several GB of models |
| `LEASE_AGENT_LEASE_STEPS`, `LEASE_AGENT_ISSUE_STEPS` | Change which steps each agent runs, and in what order (comma-separated names) | nothing; see [Editing the pipeline](#editing-the-pipeline) |

Install the provider SDKs with `pip install -r requirements-vision.txt`. A wrong provider name, a missing
key or a missing SDK stops the app at startup with a clear message. A provider failure at run time (bad
key, rate limit, timeout) returns a 502 that names the step, and nothing is half-saved.

**Try a provider on real photos before relying on it:** start the app with the provider and report an
issue with a real photo.

```bash
export OPENAI_API_KEY=...          # or ANTHROPIC_API_KEY, set in your own shell, never in a file
LEASE_AGENT_VISION=openai uvicorn app.main:app      # or anthropic
```

Open a unit and upload a photo. Use real photos: the placeholders in `samples/photos` have their own file
names written on them, so a model would just read them back.

The **vision provider adapters have not been run against a live service in this repo**. Request shape
and reply parsing are tested with fake clients, but a real call can still differ, so try it with a real
photo first.

**Test the OCR on a scan:**

```bash
python3.11 -m venv .venv-ocr && source .venv-ocr/bin/activate     # Python 3.11 or 3.12, not 3.14
pip install -r requirements-ocr.txt
LEASE_AGENT_OCR=paddle uvicorn app.main:app                        # then upload the scan in the UI
```

Two scanned samples, both with no text layer:

| Sample | What it is | Result on the author's machine (Intel Mac, no CUDA, CPU) |
|---|---|---|
| `samples/lease_scanned.pdf` | A clean render of the lease | About 30 s. All 16 fields right, all 7 rules pass, 0 flags. [Picture](docs/ocr-clean.jpg) |
| `samples/lease_scanned_photo.pdf` | Imitates a phone photo: crooked, blurred, uneven light, noise | All 16 fields right, all 7 rules pass, 2 flags. [Picture](docs/ocr-phone-photo.jpg) |

The photo sample exposed three real misreads, now fixed and covered by tests (`tests/test_ocr_misreads.py`):
OCR turned `9,500` into `9.500` (first read as 9.5), dropped a letter from "Security Deposit", and ran
`5. RENT` together as `5.RENT`. A period where a thousands comma belongs is now read as thousands **and
flagged** ("the page shows 9.500, confirm against the page"), because a guess like that must not be trusted
silently. The samples are synthetic, so they are still easier than a real crumpled scan. Test a real phone
photo of a printed lease, and an Arabic one, before trusting it.

## How it works

**Watch it run:** open [`docs/how-it-works.html`](docs/how-it-works.html) in a browser (no server needed).
It replays the real pipeline step by step for five cases: a clean lease, a lease with problems, a scan
with no OCR, a clear photo and an unclear photo. Every line it shows comes from running the code on the
sample files (`python -m scripts.export_trace` regenerates it), so it cannot drift from the code.

```
Part A                                            Part B
PDF ─► text + boxes (OCR only for scanned pages)  photos (+ note) ─► vision model (interface)
   ─► field extraction with provenance               ─► condition, damage, equipment per photo
   ─► self-check (missing, conflicts, odd values)    ─► draft work order (title, detail, priority)
   ─► owner rules R1-R7                                   │
   ─► flags                                               │
   ─► HUMAN: accept / reject / override                   ├─► HUMAN: accept / edit / reject
   ─► accept lease ─► unit occupied                       │
                      └────────── both attach to the unit ┘
                       GET /api/units/{id}: lease + issues + audit, one screen
```

### Key decisions

**1. A model only where it earns its place.** Leases are regular legal prose, and the rules are
arithmetic. So the lease side has no LLM: text-layer extraction, regex clauses, and a deterministic
rule engine. That makes results repeatable, cheap, testable, and able to cite the exact clause. A
pattern that fails returns an empty field that gets flagged, instead of a plausible guess. Reading a
photo is the one step that genuinely needs a model, so it sits behind a `VisionModel` interface
(`app/vision.py`). The same reasoning drives the parser: a wrong rent or date is a real cost to
someone, so loud failure beats fluent hallucination.

**2. Agents as loops with a human gate, not a form with an LLM attached.** `LeaseAgent` runs read,
extract, self-check, validate, flag, then waits. A human override re-runs validation and re-derives
the flags, and decisions already made on unchanged flags survive (flag ids are stable, `kind.field`).
`IssueAgent` assesses each photo, merges findings and drafts, and never dispatches anything.
Each run records a step trace that the UI shows.

**3. Traceability is the data model.** Every `ExtractedField` carries `{page, bbox, quote}`,
a confidence and a decision (`pending / accepted / rejected / overridden`), and keeps the agent's
original value after an override. The UI draws the source box on the page image. Rejected fields are
treated as missing by the rules, so rejecting a value cannot silently pass a rule.

**4. OCR only when needed, and the best open one when it is.** Digital PDFs already carry text and
coordinates, so they never touch OCR. A page without a text layer goes to an `OcrEngine`. I surveyed
2026 benchmarks and picked **PaddleOCR-VL-1.6** as the best open model (about 0.9B parameters,
Apache-2.0, 96.34 on OmniDocBench v1.6, first on the scanned Real5 subset; MinerU2.5-Pro scored 95.75
and GLM-OCR 95.22). Sources:
[Roboflow ranking](https://blog.roboflow.com/best-open-source-ocr-models/),
[Docsumo comparison](https://www.docsumo.com/blog/best-ocr-models),
[OmniDocBench](https://github.com/opendatalab/OmniDocBench).

**What actually ran.** When I tried it, PaddleOCR-VL would not load on an Intel Mac: it needs
PaddlePaddle 3.2 or newer, and only 3.0.0 is published for that platform. So the engine verified end to
end is **classic PP-OCR** from the same project (`paddle`), which returns line-level boxes, runs on CPU,
and gave a clean read of the scanned sample. `paddle-vl` is still in the code for Linux or Apple silicon,
and fails with a clear message when the install is too old. Both sit behind the same interface. A few
leaderboard points are not a reason to lock in, and neither has been run on real Doha leases.

**5. The unit is the join.** Leases and issues both reference `unit_id`. `GET /api/units/{id}`
returns the accepted lease, leases awaiting review, all issues with their work orders, and the audit
trail. Unit status is an overlay in the database over the supplied `units.json`, so the seed file
stays untouched and occupancy changes only when a person accepts a lease.

**6. The ruleset drives the engine; code supplies one function per rule id.** The file decides which
rules run and their severity. I did not `eval` the `check` strings. A rule id with no function reports
NOT_DETERMINABLE, so adding a rule to the JSON can never silently pass.

### Editing the pipeline

An agent is a list of registered **steps**, not a hard-coded function. The defaults are:

```
lease:  read_pdf → extract_fields → match_unit → validate_rules → self_check
issue:  assess_photos → summarise → draft_work_order
```

Each step does one job and declares what it `needs` and what it `provides`. `GET /api/pipelines` shows the
active lists and every step that could be added.

| I want to | Do this |
|---|---|
| Drop a step (assessment-only issue agent, no flags) | `LEASE_AGENT_ISSUE_STEPS=assess_photos,summarise` |
| Reorder or change engines | Set the step list; set `LEASE_AGENT_OCR` / `LEASE_AGENT_VISION` |
| Add a task (market-rent check, vendor routing, an Arabic translation step) | Write one class, register it, add its name to the list |

```python
@register("lease", "check_market_rent")
class CheckMarketRent(BaseStep):
    """Flag rent far above the going rate for the building."""
    needs = frozenset({"fields", "flags"})
    provides = frozenset({"market_check"})
    rerun = True                       # run again after a human override
    def run(self, ctx): ...            # read ctx.lease, append a Flag
```

What keeps this safe to edit:

- **A bad list fails when the app starts,** not on the first upload. An unknown name, a step placed before
  what it needs, or a missing service (a step that needs OCR with none configured) raises `PipelineError`
  listing what is available.
- **A step that fails is named** (`lease step 'check_market_rent' failed: ...`) and the upload returns a clear 422.
- **`rerun` steps are the deterministic ones.** After a human overrides a field, only steps marked `rerun`
  run again (`match_unit`, `validate_rules`, `self_check`). The PDF is not re-read, and decisions already
  made on unchanged flags survive.
- **The human gate stays outside the pipeline.** Accept, reject and override are separate API calls, and the
  accept endpoint enforces its own checks (a matched, available unit; every high-severity flag decided)
  whatever steps ran before it. The shipped steps never set a lease's status. A custom step could, so
  review new steps like any code that touches decisions.

The step lists live in code (`DEFAULT_LEASE_STEPS`, `DEFAULT_ISSUE_STEPS`) with environment overrides.
A YAML pipeline file with per-owner step options is the obvious next step, and the registry is what it
would be built on. I stopped short of it because the brief asks for a small service, and an untested config
language would be more surface than value.

### Decisions you might disagree with

- **Term vs dates (R4)** accepts both end-date conventions: 1 Mar 2026 to 28 Feb 2027 and to 1 Mar 2027
  both count as 12 months.
- **Absence of evidence.** On a document that could not be read at all, "no escalation clause" and
  "parties not identified" are NOT_DETERMINABLE, not FAIL.
- **Accepting a lease** requires: a matched, available unit; no undecided high-severity flag. Lower
  severity flags do not block.
- **Signatures** are judged from the text layer (letters after the signature label). A handwritten
  signature on a scan is not verified, and the field says so.
- **Dates** like `01/03/2026` are read day-first.

## What I left out

- Authentication, roles and multi-tenancy. Everything is one owner, one trust boundary.
- Image-based leases other than PDF; multi-document leases (amendments, addenda, annexes).
- Arabic and bilingual leases. Common in Doha, and the first thing I would test, but the clause
  patterns here are English only.
- Verifying handwritten signatures or stamps visually.
- Dispatching work orders to vendors, tenant notifications, comments.
- Real sample photos: the brief's photos were not supplied, so the images here are placeholders and the
  stub reads their names. The vision path is the least validated part of this repo.
- Rent schedule, cheques and payments.

## Where it breaks first at scale

1. **Synchronous processing.** OCR and vision calls run inside the request. Move to a job queue with
   status polling before the first 50-page scan.
2. **SQLite plus one lock.** Fine for a demo, a single writer in production. The `Store` class is the
   only place that knows, so swapping in Postgres is one class. Leases and issues are JSON documents,
   and `list_leases()` scans every lease, which the unit view calls. Needs real columns and indexes.
3. **Local file storage** for PDFs and photos. Needs object storage and retention rules.
4. **Pattern coverage.** The regex parser is exact on the clause layouts it knows. Each new lease template
   is a new pattern. The planned answer is a model that fills **only** the fields the parser left empty,
   still carrying a source quote that the code verifies exists in the document, and with every one of
   them sent to review. It is deliberately not built yet.
5. **Ruleset growth.** Per-rule Python is clear for 7 rules. At 70, per-owner and versioned, it wants a
   small declarative rule format.
6. **Unbounded audit and no review queue.** Reasonable at ten leases, not at ten thousand.

## Product enhancement ideas

I would build these roughly in this order, and the first three come from the same observation: the unit
is the join between the lease and what happens in the unit, and nobody is using that yet.

1. **Let the lease decide who pays.** Extract the maintenance and responsibility clauses and, when an
   issue is reported, say whether it looks like an owner or a tenant cost, quoting the clause. Today the
   two features share a screen; this makes them share a decision. It turns a work order from "something
   is broken" into "owner's cost, vendor needed, here is the clause".
2. **Move-in and move-out condition baselines.** Photos at handover become the baseline for the unit.
   At move-out the agent diffs against it and drafts a deposit-deduction case with before and after
   evidence. Deposit disputes are where a wrong condition call costs real money, and it reuses Part B
   unchanged.
3. **Equipment registry per unit.** Part B already lists the AC, heater and appliances it sees. Keep them
   as assets with age and warranty, so a repeat leak on the same AC becomes "third call in four months,
   replace it" instead of three unrelated tickets. In construction terms this is also a defects-liability
   tracker: whether a defect is still inside the contractor's warranty window.
4. **An obligations calendar from the lease.** Expiry, renewal-notice deadlines, escalation dates, and
   the early-termination notice period are already extracted. Turn them into dated reminders so the owner
   acts 90 days before a lease ends instead of finding out at expiry.
5. **Review by exception.** Every override is a labelled example. Track the override rate per field, and
   let leases that pass all rules with high-confidence fields go to a quick one-click review while risky
   ones get the full screen. The aim is to shrink review time without removing the human, and the override
   rate is the number that says whether it is safe.
6. **Tenant intake where tenants already are.** A WhatsApp-style photo intake with a follow-up question
   ("is the water still coming out?") feeds the same issue agent, which is how a stub becomes a product
   people use.
7. **Per-owner rule packs, with a simulator.** Let an owner edit rules in plain language, and replay them
   over past leases ("how many existing leases would this new 24-month cap have flagged?") before saving.
8. **Portfolio view.** Occupancy, leases expiring in 90 days, open work orders by priority, and rent roll.

Measures I would watch from day one: time from upload to accepted record, override rate per field,
share of work orders accepted unedited, and the rate of "confirmed" flags that were real.

## Layout

```
app/
  domain.py    models: Field + provenance, Flag, RuleResult, Lease, Issue, WorkOrder
  ingest.py    PDF to positioned lines; OcrEngine interface and routing
  extract.py   rule-based field extraction
  rules.py     R1-R7 engine
  units.py     unit registry and matching
  pipeline.py  Step, registry, Pipeline: validates the step list, runs it, names failures
  lease_steps.py  read_pdf, extract_fields, match_unit, validate_rules, self_check
  issue_steps.py  assess_photos, summarise, draft_work_order
  agents.py    LeaseAgent, IssueAgent (a step list each) and human-decision handling
  vision.py    VisionModel interface, StubVision, AnthropicVision, OpenAIVision
  ocr_paddle.py optional OCR engines: PP-OCR (verified) and PaddleOCR-VL (not run here)
  store.py     SQLite persistence and audit log
  main.py      FastAPI app
  static/      single-page UI
seed/          units.json and owner_ruleset.json as supplied
scripts/       sample generator
tests/         145 tests
```

## Honest status

Tested: the rule engine, extraction and provenance, OCR routing, unit matching, the whole review flow over
HTTP, input validation, the OCR adapters and both vision adapters against fakes. **Run for real:** PP-OCR on
the scanned sample, both directly and through an HTTP upload to the app (fields, rules, source boxes and
accepting the lease all correct). Not run live: PaddleOCR-VL (it cannot load on the author's machine) and the
Anthropic and OpenAI vision calls. The UI is exercised only by hand. Photo assessment from the stub is
keyword-driven by design.
