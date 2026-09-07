# 📖 فن اللامبالاة — Automated Daily Motivational Video Pipeline

An end-to-end, fully automated, serverless pipeline that generates one new **~60-second vertical motivational video** per day, distilling key lessons from Mark Manson's *"The Subtle Art of Not Giving a F\*ck"* — complete with vibrant AI-generated imagery, Arabic voiceover, TikTok-ready caption, and hashtags — and delivers it to a Telegram chat for manual posting.

> ⚡ **Content Style**: Motivational & Encouraging — NOT storytelling. Each episode speaks DIRECTLY to the viewer as a life coach would, covering one key lesson from the book.

> 🎨 **Visual Style**: Ultra-vibrant, high-saturation, modern, and bright images that energize and inspire — golden hour cityscapes, neon aesthetics, bold geometric art.

The series continues automatically: each run reads [`book_state.json`](book_state.json) to see which lessons have already been covered, picks the next one, and commits the updated state back to the repo.

---

## 🏗️ Architecture Overview

```
GitHub Actions (Daily Cron / Manual)
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│                          main.py                                │
│                                                                  │
│  Step 0 │ book_state.json    → Load last episode + lesson summary│
│  Step 1 │ Claude (Anthropic) → Motivational Script + 4 Vibrant  │
│         │                      Image Prompts + Caption + Hashtags│
│  Step 2 │ Edge-TTS           → Arabic MP3 Voiceover              │
│  Step 3 │ Pollinations.ai    → 4 × 1080×1920 Vibrant Images      │
│  Step 4 │ MoviePy + FFmpeg   → final_video.mp4                   │
│  Step 5 │ Telegram Bot API   → Video + Caption + Hashtags        │
│  Step 6 │ book_state.json    → Save lesson progress (git commit) │
└────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
.
├── main.py                        # Core pipeline application
├── book_state.json                # Episode number + running lesson summary
├── requirements.txt               # Python dependencies
├── .github/
│   └── workflows/
│       └── automate.yml           # GitHub Actions CI/CD workflow
└── README.md                      # This file
```

---

## ⚙️ Setup Instructions

### Prerequisites

- A GitHub repository (this one ✅)
- An Anthropic account (for Claude API access)
- A Telegram Bot created via [@BotFather](https://t.me/BotFather)
- The bot added to your target channel/group with admin rights

---

## 🔐 Required GitHub Secrets

Configure **3 repository secrets** in: **Settings → Secrets and variables → Actions**

| Secret Name | Description |
|-------------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key from [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| `TELEGRAM_BOT_TOKEN` | Bot token from [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | Your channel/group ID (e.g., `@mychannel` or `-1001234567890`) |

---

## ▶️ Running the Pipeline

### Automatic (Scheduled)
Runs **once daily at 10:00 UTC** (1:00 PM Riyadh Time), producing one new motivational episode.

### Manual Trigger
1. Go to your repository → **Actions** tab.
2. Select **"📖 فن اللامبالاة - Automated Video Pipeline"**.
3. Click **"Run workflow"**.

---

## 📖 Lesson Continuity (`book_state.json`)

```json
{
  "episode": 5,
  "summary": "Episodes 1-5 covered: the concept of not giving a f*ck, choosing values wisely, taking responsibility, the importance of saying no, and embracing uncertainty."
}
```

- Each run reads this file to know which lessons are already covered, so no lesson is repeated.
- After a successful run, the workflow commits the updated state back to the repo.
- To restart from the beginning: delete `book_state.json` or reset it to `{"episode": 0, "summary": ""}`.

---

## 🎨 What Makes This Different

| Feature | This Project |
|---------|-------------|
| Content type | Motivational life-coach style |
| Speaks to viewer | Direct ("أنت") — not third-person narration |
| Image style | Ultra-vibrant, neon, bright, energizing |
| Book coverage | Key lessons from "The Subtle Art of Not Giving a F\*ck" |
| Schedule | Daily at 10:00 UTC (offset from sibling Kafka project) |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `anthropic` | Claude API for script & prompt generation |
| `edge-tts` | Microsoft Edge TTS for Arabic voiceover |
| `moviepy` | Video assembly (images + audio → MP4) |
| `requests` | Image fetching & Telegram API delivery |
| `Pillow` | Image processing backend for MoviePy |
| `ffmpeg` | System-level video encoding (installed via apt) |

---

## 📄 License

Personal/educational use. Ensure compliance with Anthropic Claude, Edge-TTS, and Pollinations.ai Terms of Service.
