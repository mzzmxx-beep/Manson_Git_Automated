# -*- coding: utf-8 -*-
"""
Automated Vertical Video Generation Pipeline - فن اللامبالاة
=============================================================
Book: "The Subtle Art of Not Giving a F*ck" by Mark Manson
Content Style: Motivational / Self-improvement (NOT storytelling)

Pipeline:
  1. Generate today's episode key lesson, motivational script, vibrant visual
     prompts, TikTok caption, hashtags, and a running summary via Claude
  2. Generate Arabic voiceover (TTS) via Edge-TTS
  3. Fetch 4 AI-generated vibrant images from Pollinations.ai
  4. Assemble images + audio into a 9:16 MP4 via MoviePy
  5. Send the final video (with caption + hashtags) to a Telegram chat
     for manual posting to TikTok
  6. Persist episode progress to book_state.json so the next run
     covers the next lesson instead of repeating it

Environment Variables Required:
  - ANTHROPIC_API_KEY   : Anthropic API key
  - TELEGRAM_BOT_TOKEN  : Telegram Bot token from @BotFather
  - TELEGRAM_CHAT_ID    : Target Telegram chat/channel ID
"""

import json
import os
import sys
import asyncio
import logging
import random
import re
import time
import urllib.parse

import requests
from anthropic import Anthropic
import edge_tts
import numpy as np
from moviepy.editor import (
    ImageClip,
    AudioFileClip,
    concatenate_videoclips,
    VideoClip,
    CompositeVideoClip,
)
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30
TTS_VOICE = "ar-SA-HamedNeural"
AUDIO_OUTPUT = "voiceover.mp3"
VIDEO_OUTPUT = "final_video.mp4"
NUM_IMAGES = 4
POLLINATIONS_BASE = (
    "https://image.pollinations.ai/prompt/{prompt}"
    "?width=1080&height=1920&nologo=true&model=flux&seed={seed}&enhance=true"
)
STATE_FILE = "book_state.json"
BOOK_TITLE = "فن اللامبالاة"
BOOK_AUTHOR = "مارك مانسون"

# ---------------------------------------------------------------------------
# Claude prompt template — Motivational content style (NOT storytelling)
# ---------------------------------------------------------------------------
CLAUDE_PROMPT_TEMPLATE = (
    "You are an expert Arabic motivational content creator and life-coach\n"
    "producing a daily short-video series based on the internationally\n"
    "acclaimed self-help book 'The Subtle Art of Not Giving a F*ck'\n"
    "by Mark Manson, for TikTok/Reels (9:16 vertical, ~60 seconds).\n\n"
    "This is episode number {episode_number} of the series.\n"
    "Lessons already covered (empty if this is the first episode):\n"
    '"""\n{previous_summary}\n"""\n\n'
    "Task:\n"
    "1. Choose the NEXT KEY LESSON or concept from the book that has NOT\n"
    "   been covered yet according to the summary above.\n"
    "   Write a powerful, energetic, motivational Arabic script (Modern\n"
    "   Standard Arabic with some relatable colloquial touches).\n"
    "   The tone must be:\n"
    "     - Encouraging, empowering, and direct (like a life coach speaking)\n"
    "     - NOT a story narration — speak directly TO the viewer\n"
    "     - Use short punchy sentences with rhythm and impact\n"
    "     - Include a memorable takeaway or call-to-action at the end\n"
    "   Keep it under 150 words so it fits in ~60 seconds when narrated.\n\n"
    "2. After the script, provide exactly 4 distinct English visual prompts\n"
    "   for AI image generation. Each prompt MUST describe a VIBRANT,\n"
    "   MODERN, BRIGHT, UPLIFTING scene that visually represents the\n"
    "   lesson's concept. Style requirements for ALL images:\n"
    "     - Ultra-bright, high-saturation color palette\n"
    "     - Modern minimalist or neon aesthetic\n"
    "     - Motivational and inspirational atmosphere\n"
    "     - Contemporary lifestyle or abstract visualization\n"
    "     - No dark or gloomy elements — pure energy and positivity\n"
    "     - 9:16 vertical format, cinematic quality, photorealistic\n"
    "   Example styles: golden hour cityscapes, vibrant geometric art,\n"
    "   neon-lit modern interiors, energetic athletic scenes, sunrise\n"
    "   mountain peaks, glowing abstract particles, bold typography art.\n\n"
    "3. Write a short, punchy Arabic caption (max 2 sentences) for the\n"
    "   TikTok post: a bold motivational statement that makes people STOP scrolling.\n\n"
    "4. Provide 7-10 relevant hashtags (mix of Arabic and English),\n"
    "   space-separated, each starting with #.\n\n"
    "5. Write a concise 1-2 sentence summary of WHICH LESSON was covered\n"
    "   in this episode, for continuity context in the next run.\n\n"
    "Format your response EXACTLY as follows (do not deviate):\n\n"
    "---SCRIPT---\n"
    "[Arabic motivational script here]\n\n"
    "---PROMPTS---\n"
    "1. [First English visual prompt]\n"
    "2. [Second English visual prompt]\n"
    "3. [Third English visual prompt]\n"
    "4. [Fourth English visual prompt]\n\n"
    "---CAPTION---\n"
    "[Arabic motivational caption here]\n\n"
    "---HASHTAGS---\n"
    "[hashtags here]\n\n"
    "---SUMMARY---\n"
    "[lesson summary here]\n"
)


# ===========================================================================
# BOOK STATE - Track episode/lesson progress across daily runs
# ===========================================================================
def load_book_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        logger.info(
            "Loaded book state: episode {}.".format(state.get("episode", 0) + 1)
        )
        return state
    logger.info("No existing book state found. Starting from episode 1.")
    return {"episode": 0, "summary": ""}


def save_book_state(episode_number, summary):
    state = {"episode": episode_number, "summary": summary}
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    logger.info("Saved book state: episode {}.".format(episode_number))


# ===========================================================================
# STEP 1 - Script & Visual Prompts (Anthropic Claude)
# ===========================================================================
def generate_script_and_prompts(episode_number, previous_summary):
    logger.info("=== STEP 1: Generating episode {} via Claude ===".format(episode_number))

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")

    client = Anthropic(api_key=api_key)
    CLAUDE_MODEL = "claude-sonnet-5"

    prompt = CLAUDE_PROMPT_TEMPLATE.format(
        episode_number=episode_number,
        previous_summary=previous_summary or "(this is the first episode — start from the very beginning of the book)",
    )

    raw_text = None
    for attempt in range(1, 4):
        try:
            logger.info("Attempt {}/3 with model {}...".format(attempt, CLAUDE_MODEL))
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1536,
                messages=[{"role": "user", "content": prompt}],
            )
            text_blocks = [b.text for b in response.content if b.type == "text"]
            if not text_blocks:
                raise ValueError("No text block found in Claude response.")
            raw_text = "".join(text_blocks)
            logger.info("Claude response received from {}.".format(CLAUDE_MODEL))
            break
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower() or "overloaded" in err_str.lower():
                wait_sec = 65 * attempt
                logger.warning(
                    "Rate-limit/overload on attempt {}/3. Waiting {}s...".format(attempt, wait_sec)
                )
                time.sleep(wait_sec)
            else:
                logger.error("Claude error: {}".format(e))
                raise

    if not raw_text:
        raise RuntimeError("Claude API exhausted after retries. Check API key and rate limits.")

    script_match = re.search(r"---SCRIPT---\s*(.*?)\s*---PROMPTS---", raw_text, re.DOTALL)
    if not script_match:
        raise ValueError("Could not parse script from Claude response.")
    script = script_match.group(1).strip()

    prompts_match = re.search(r"---PROMPTS---\s*(.*?)\s*---CAPTION---", raw_text, re.DOTALL)
    if not prompts_match:
        raise ValueError("Could not parse prompts from Claude response.")
    prompts_raw = prompts_match.group(1).strip()

    prompts = []
    for line in prompts_raw.splitlines():
        line = line.strip()
        match = re.match(r"^\d+[\.\)]\s+(.+)$", line)
        if match:
            prompts.append(match.group(1).strip())

    if len(prompts) < NUM_IMAGES:
        raise ValueError(
            "Expected {} visual prompts, got {}. Raw:\n{}".format(NUM_IMAGES, len(prompts), prompts_raw)
        )

    caption_match = re.search(r"---CAPTION---\s*(.*?)\s*---HASHTAGS---", raw_text, re.DOTALL)
    if not caption_match:
        raise ValueError("Could not parse caption from Claude response.")
    caption = caption_match.group(1).strip()

    hashtags_match = re.search(r"---HASHTAGS---\s*(.*?)\s*---SUMMARY---", raw_text, re.DOTALL)
    if not hashtags_match:
        raise ValueError("Could not parse hashtags from Claude response.")
    hashtags = hashtags_match.group(1).strip()

    summary_match = re.search(r"---SUMMARY---\s*(.*)", raw_text, re.DOTALL)
    if not summary_match:
        raise ValueError("Could not parse episode summary from Claude response.")
    episode_summary = summary_match.group(1).strip()

    logger.info("Script parsed successfully ({} chars).".format(len(script)))
    logger.info("Extracted {} visual prompts.".format(len(prompts)))
    return {
        "script": script,
        "prompts": prompts[:NUM_IMAGES],
        "caption": caption,
        "hashtags": hashtags,
        "summary": episode_summary,
    }


# ===========================================================================
# STEP 2 - Text-to-Speech / Arabic Voiceover (Edge-TTS)
# ===========================================================================
async def _generate_tts_async(script, output_path):
    communicate = edge_tts.Communicate(text=script, voice=TTS_VOICE)
    await communicate.save(output_path)


def generate_voiceover(script):
    logger.info("=== STEP 2: Generating Arabic voiceover via Edge-TTS ===")
    try:
        asyncio.run(_generate_tts_async(script, AUDIO_OUTPUT))
        if not os.path.exists(AUDIO_OUTPUT):
            raise FileNotFoundError("TTS output file not found: {}".format(AUDIO_OUTPUT))
        size_kb = os.path.getsize(AUDIO_OUTPUT) / 1024
        logger.info("Voiceover saved: {} ({:.1f} KB)".format(AUDIO_OUTPUT, size_kb))
    except Exception as e:
        logger.error("TTS generation failed: {}".format(e))
        raise


# ===========================================================================
# STEP 3 - Image Generation (Pollinations.ai) - Vibrant & Modern
# ===========================================================================
def generate_images(prompts):
    logger.info("=== STEP 3: Fetching vibrant images from Pollinations.ai ===")
    image_paths = []

    STYLE_SUFFIX = (
        ", ultra vibrant colors, high saturation, bright modern aesthetic, "
        "energetic atmosphere, photorealistic, 8K quality, professional photography, "
        "golden hour lighting, dynamic composition"
    )

    for idx, prompt in enumerate(prompts):
        filename = "image_{}.jpg".format(idx + 1)
        boosted_prompt = prompt + STYLE_SUFFIX
        encoded_prompt = urllib.parse.quote(boosted_prompt)
        seed = random.randint(1, 999999)
        url = POLLINATIONS_BASE.format(prompt=encoded_prompt, seed=seed)

        logger.info("Fetching image {}/{}: {}...".format(idx + 1, NUM_IMAGES, prompt[:60]))
        retries = 3

        for attempt in range(1, retries + 1):
            try:
                response = requests.get(url, timeout=90)
                response.raise_for_status()
                with open(filename, "wb") as f:
                    f.write(response.content)
                size_kb = os.path.getsize(filename) / 1024
                logger.info("  Image {} saved: {} ({:.1f} KB)".format(idx + 1, filename, size_kb))
                image_paths.append(filename)
                break
            except requests.RequestException as e:
                logger.warning("  Attempt {}/{} failed: {}".format(attempt, retries, e))
                if attempt < retries:
                    wait_sec = 5 * attempt
                    logger.info("  Retrying in {}s...".format(wait_sec))
                    time.sleep(wait_sec)
                else:
                    logger.error("  All {} attempts failed for image {}.".format(retries, idx + 1))
                    raise

    logger.info("All {} images fetched successfully.".format(len(image_paths)))
    return image_paths


# ===========================================================================
# STEP 4 - Video Assembly (MoviePy)
# ===========================================================================
def _make_ken_burns_frame(img_array, effect_type, duration, zoom_factor, w, h):
    def make_frame(t):
        progress = t / max(duration, 1e-6)
        if effect_type == "zoom_in":
            scale = 1.0 + (zoom_factor - 1.0) * progress
            cx, cy = 0.0, 0.0
        elif effect_type == "zoom_out":
            scale = zoom_factor - (zoom_factor - 1.0) * progress
            cx, cy = 0.0, 0.0
        elif effect_type == "pan_right":
            scale = zoom_factor
            cx = (zoom_factor - 1.0) * w * progress
            cy = 0.0
        elif effect_type == "pan_left":
            scale = zoom_factor
            cx = -(zoom_factor - 1.0) * w * progress
            cy = 0.0
        elif effect_type == "pan_up":
            scale = zoom_factor
            cx = 0.0
            cy = -(zoom_factor - 1.0) * h * progress
        else:
            scale = zoom_factor
            cx = 0.0
            cy = (zoom_factor - 1.0) * h * progress

        new_w = max(w, int(w * scale))
        new_h = max(h, int(h * scale))
        scaled = PILImage.fromarray(img_array).resize((new_w, new_h), PILImage.LANCZOS)
        scaled_arr = np.array(scaled)
        base_x = (new_w - w) // 2
        base_y = (new_h - h) // 2
        x_start = int(base_x + cx)
        y_start = int(base_y + cy)
        x_start = max(0, min(x_start, new_w - w))
        y_start = max(0, min(y_start, new_h - h))
        return scaled_arr[y_start : y_start + h, x_start : x_start + w]

    return make_frame


def assemble_video(image_paths, audio_path):
    logger.info("=== STEP 4: Assembling video with MoviePy (Ken Burns + Crossfade) ===")
    FADE_DURATION = 0.5
    ZOOM_FACTOR = 1.12
    KB_EFFECTS = ["zoom_in", "zoom_out", "pan_right", "pan_left", "pan_up", "pan_down"]

    try:
        audio_clip = AudioFileClip(audio_path)
        total_duration = audio_clip.duration
        logger.info("Audio duration: {:.2f}s".format(total_duration))

        n = len(image_paths)
        per_image_duration = (total_duration + (n - 1) * FADE_DURATION) / n
        logger.info(
            "Each image will display for {:.2f}s ({} images, {:.1f}s crossfade)".format(
                per_image_duration, n, FADE_DURATION
            )
        )

        raw_clips = []
        for idx, img_path in enumerate(image_paths):
            effect = KB_EFFECTS[idx % len(KB_EFFECTS)]
            logger.info("  Building Ken Burns clip {}/{} (effect: {})".format(idx + 1, n, effect))
            pil_img = (
                PILImage.open(img_path)
                .convert("RGB")
                .resize((VIDEO_WIDTH, VIDEO_HEIGHT), PILImage.LANCZOS)
            )
            img_array = np.array(pil_img)
            make_frame = _make_ken_burns_frame(
                img_array, effect, per_image_duration, ZOOM_FACTOR, VIDEO_WIDTH, VIDEO_HEIGHT
            )
            clip = VideoClip(make_frame, duration=per_image_duration)
            raw_clips.append(clip)

        logger.info("Applying crossfade transitions ({:.1f}s each)...".format(FADE_DURATION))
        positioned_clips = []
        current_start = 0.0
        for i, clip in enumerate(raw_clips):
            if i == 0:
                placed = clip.set_start(0)
            else:
                placed = clip.crossfadein(FADE_DURATION).set_start(current_start)
            positioned_clips.append(placed)
            current_start += clip.duration - FADE_DURATION

        final_clip = (
            CompositeVideoClip(positioned_clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT))
            .set_duration(total_duration)
            .set_audio(audio_clip)
        )

        logger.info(
            "Exporting video: {} at {} fps (CRF=18, 8 Mbps, slow preset)...".format(VIDEO_OUTPUT, VIDEO_FPS)
        )
        final_clip.write_videofile(
            VIDEO_OUTPUT,
            fps=VIDEO_FPS,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            temp_audiofile="temp_audio.m4a",
            remove_temp=True,
            ffmpeg_params=["-crf", "18", "-preset", "slow", "-b:v", "8M"],
            logger=None,
        )

        audio_clip.close()
        final_clip.close()
        for clip in raw_clips:
            clip.close()

        size_mb = os.path.getsize(VIDEO_OUTPUT) / (1024 * 1024)
        logger.info("Video exported successfully: {} ({:.2f} MB)".format(VIDEO_OUTPUT, size_mb))

    except Exception as e:
        logger.error("Video assembly failed: {}".format(e))
        raise


# ===========================================================================
# STEP 5 - Telegram Delivery
# ===========================================================================
def send_to_telegram(video_path, episode_number, tiktok_caption, hashtags):
    logger.info("=== STEP 5: Sending video to Telegram ===")

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token:
        raise EnvironmentError("TELEGRAM_BOT_TOKEN environment variable is not set.")
    if not chat_id:
        raise EnvironmentError("TELEGRAM_CHAT_ID environment variable is not set.")
    if not os.path.exists(video_path):
        raise FileNotFoundError("Video file not found: {}".format(video_path))

    api_url = "https://api.telegram.org/bot{}/sendVideo".format(bot_token)
    caption = (
        "📖 {book} — {author} | حلقة {episode}\n\n"
        "{tiktok_caption}\n\n"
        "{hashtags}"
    ).format(
        book=BOOK_TITLE,
        author=BOOK_AUTHOR,
        episode=episode_number,
        tiktok_caption=tiktok_caption,
        hashtags=hashtags,
    )

    try:
        with open(video_path, "rb") as video_file:
            logger.info("Uploading {} to Telegram...".format(video_path))
            response = requests.post(
                api_url,
                data={"chat_id": chat_id, "caption": caption},
                files={"video": video_file},
                timeout=120,
            )

        response.raise_for_status()
        result = response.json()

        if result.get("ok"):
            logger.info("Video delivered to Telegram successfully!")
        else:
            raise RuntimeError("Telegram API error: {}".format(result))

    except Exception as e:
        logger.error("Telegram delivery failed: {}".format(e))
        raise


# ===========================================================================
# CLEANUP
# ===========================================================================
def cleanup_temp_files(image_paths):
    logger.info("=== Cleaning up temporary files ===")
    files_to_remove = list(image_paths) + [AUDIO_OUTPUT, VIDEO_OUTPUT]
    for filepath in files_to_remove:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info("  Removed: {}".format(filepath))
            except OSError as e:
                logger.warning("  Could not remove {}: {}".format(filepath, e))


# ===========================================================================
# MAIN ENTRYPOINT
# ===========================================================================
def main():
    logger.info("=" * 70)
    logger.info("  فن اللامبالاة - AUTOMATED VIDEO GENERATION PIPELINE - STARTING")
    logger.info("=" * 70)

    image_paths = []

    try:
        state = load_book_state()
        episode_number = state.get("episode", 0) + 1
        previous_summary = state.get("summary", "")

        episode = generate_script_and_prompts(episode_number, previous_summary)
        generate_voiceover(episode["script"])
        image_paths = generate_images(episode["prompts"])
        assemble_video(image_paths, AUDIO_OUTPUT)
        send_to_telegram(
            VIDEO_OUTPUT, episode_number, episode["caption"], episode["hashtags"]
        )
        save_book_state(episode_number, episode["summary"])

        logger.info("=" * 70)
        logger.info("  PIPELINE COMPLETED SUCCESSFULLY (Episode {})".format(episode_number))
        logger.info("=" * 70)

    except Exception as e:
        logger.critical("PIPELINE FAILED: {}".format(e), exc_info=True)
        sys.exit(1)

    finally:
        cleanup_temp_files(image_paths)


if __name__ == "__main__":
    main()
