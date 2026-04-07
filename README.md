# 📬 Email Open Environment (Email-OpenEnv)

> An OpenEnv-compatible AI benchmark environment where a language model agent reads, classifies, extracts information from, and generates professional replies to realistic emails — fully scored by an automated grader.

---

## 🎯 What Is This?

**Email-OpenEnv** is a structured AI evaluation environment built on the **OpenEnv** framework pattern. It simulates a real-world email inbox where an AI agent must:

| Task                  | Description                                         | Score |
| --------------------- | --------------------------------------------------- | ----- |
| 📂 Classification     | Label email: Work / Personal / Spam / Urgent        | 0–1   |
| 🔍 Extraction         | Extract name, date, deadline, request, org, contact | 0–1   |
| ✉️ Reply Generation   | Write a professional contextual reply               | 0–1   |
| 🚦 Priority Detection | Assign Low / Medium / High / Critical               | 0–1   |

**Total reward per email: 4.0 · 10 emails · Max total: 40.0**

---

## 📁 Project Structure

```
email_openenv/
│
├── app.py            # Gradio web UI
├── environment.py    # Core OpenEnv API (reset / step / state)
├── baseline.py       # GPT-4o-mini baseline agent + runner
├── models.py         # Pydantic schemas for all data types
├── grader.py         # Rule-based scoring system
├── openenv.yaml      # Environment configuration file
├── requirements.txt  # Python dependencies
├── Dockerfile        # Docker deployment
└── README.md         # This file
```

---

## ⚙️ Setup

### 1. Clone / Download

```bash
git clone <your-repo-url>
cd email_openenv
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Your OpenAI API Key

```bash
export OPENAI_API_KEY="sk-..."        # Linux / macOS
set OPENAI_API_KEY=sk-...             # Windows CMD
$env:OPENAI_API_KEY="sk-..."          # Windows PowerShell
```

Or create a `.env` file:

```
OPENAI_API_KEY=sk-...
```

---

## 🚀 How to Run

### Option A: Gradio Web UI (Recommended)

```bash
python app.py
```

Open your browser at **http://localhost:7860**

- Enter your OpenAI API key in the field
- Click **⚡ Process Email with AI** to process one email at a time
- View classification, extracted entities, AI reply, and grader score
- Click **🔄 Reset** to start over

### Option B: Baseline Agent (Terminal)

Run the full agent through all 10 emails and print final score:

```bash
python baseline.py
```

Or pass an API key directly:

```bash
python baseline.py sk-your-key-here
```

### Option C: Python API

```python
from environment import EmailEnvironment
from baseline import BaselineAgent

env = EmailEnvironment()
agent = BaselineAgent(api_key="sk-...")

state = env.reset()

while not state.task_complete:
    email = state.current_email
    action = agent.act(email)
    result = env.step(action)
    print(f"Email {email.id} → Reward: {result.reward:.2f}/4.0")
    state = result.observation

print(f"Final Score: {env.total_reward:.2f}/{env.max_possible_reward:.1f}")
```

---

## 🐳 Docker Deployment

### Build

```bash
docker build -t email-openenv .
```

### Run

```bash
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... email-openenv
```

Open **http://localhost:7860**

---

## 🧠 OpenEnv API Reference

### `environment.reset() → EnvironmentState`

Resets the environment to the first email. Returns initial state.

### `environment.step(action: AgentAction) → StepResult`

Submits the agent's action. Grades it, advances to next email, returns reward + state.

### `environment.state() → EnvironmentState`

Returns the current state without advancing.

### AgentAction Schema

```python
{
  "email_id": int,
  "category": "Work" | "Personal" | "Spam" | "Urgent",
  "priority": "Low" | "Medium" | "High" | "Critical",
  "extracted_entities": {
    "name": str | None,
    "date": str | None,
    "deadline": str | None,
    "request": str | None,
    "organization": str | None,
    "contact": str | None
  },
  "reply": str,
  "reasoning": str | None
}
```

---

## 📧 Email Dataset

10 realistic emails covering diverse real-world scenarios:

| #   | Type              | Scenario                           |
| --- | ----------------- | ---------------------------------- |
| 1   | Work / Medium     | Job Application – Python Developer |
| 2   | Work / Medium     | Leave Application – Medical        |
| 3   | Urgent / Critical | Angry Customer – Refund Demand     |
| 4   | Work / Medium     | Meeting Schedule – Q1 Strategy     |
| 5   | Spam / Low        | Lottery Scam                       |
| 6   | Urgent / Critical | Production Server Down             |
| 7   | Personal / Low    | Family Dinner Reminder             |
| 8   | Work / Medium     | HR Performance Review Notice       |
| 9   | Work / Low        | NGO Partnership Proposal           |
| 10  | Work / High       | Invoice Payment Due                |

---

## 🧮 Grader System

### Classification Score (0–1)

- Exact match against ground-truth category label
- 1.0 = correct, 0.0 = incorrect

### Extraction Score (0–1)

- Measures how many ground-truth key entities appear in extracted fields
- Full credit for entity in extraction output
- Partial credit (0.5) if entity was in email but not extracted
- Spam emails: score based on appropriate restraint (don't hallucinate)

### Reply Score (0–1)

- **Length**: ≥150 chars → 1.0, 80–150 → 0.7, <80 → 0.3
- **Tone**: counts professional keywords (thank you, regards, please, etc.)
- **Spam handling**: must include warning words; engaging = 0.0
- **Urgency handling**: urgency words boost score for Urgent emails

### Priority Score (0–1)

- Exact match = 1.0
- Adjacent level = 0.5 (e.g., High vs Critical)
- Two+ levels off = 0.0

---

## 🗂️ Configuration

See `openenv.yaml` for full environment configuration including task definitions, schemas, grader settings, and UI configuration.

---

## 📸 Screenshots

_Run the app and capture:_

1. The email panel showing a raw email
2. The AI output panel with classification badge, extracted entities, reply
3. The grader breakdown showing scores per dimension

---

## 🤝 Contributing

1. Fork the repo
2. Add new email samples to `SAMPLE_EMAILS` in `environment.py`
3. Add corresponding ground-truth to `GROUND_TRUTH`
4. Submit a PR

---

## 📄 License

MIT License – free to use, modify, and distribute.

---

## � Authors

- Subash Kumar
- Priyadharshan
- Suriya Prakash

---

## 🙏 Acknowledgements

- [OpenAI](https://openai.com) for GPT-4o-mini
- [Gradio](https://gradio.app) for the UI framework
- [Pydantic](https://docs.pydantic.dev) for data validation
- OpenEnv framework for the environment design pattern
