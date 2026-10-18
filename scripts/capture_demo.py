"""Drive the running web app with Playwright: screenshots + a short GIF for the README.

    hr-app --checkpoint checkpoints/run/best.pt --port 7860      # in another terminal
    python scripts/capture_demo.py --url http://127.0.0.1:7860 --out docs/images

Writes app_default.png, app_random_employee.png, app_probabilities.png and demo.gif.
Needs `pip install playwright` and `playwright install chromium`; the GIF needs ffmpeg
(Playwright's own copy is used when it is installed).
"""

import argparse
import glob
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright


def ffmpeg_binary() -> str | None:
    candidates = [shutil.which("ffmpeg")] + glob.glob(
        os.path.expanduser("~/.cache/ms-playwright/ffmpeg-*/ffmpeg-linux")
    )
    return next((c for c in candidates if c and Path(c).exists()), None)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:7860")
    p.add_argument("--out", type=Path, default=Path("docs/images"))
    p.add_argument("--no-gif", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    video_dir = Path(tempfile.mkdtemp(prefix="hr-jev-demo-"))

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except PlaywrightError:  # no Playwright-managed Chromium: use the installed Chrome
            browser = pw.chromium.launch(channel="chrome")
        context = browser.new_context(
            viewport={"width": 1280, "height": 860},
            record_video_dir=None if args.no_gif else str(video_dir),
            record_video_size={"width": 1280, "height": 860},
        )
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(args.url, wait_until="load")
        page.wait_for_selector("text=Random test employee", timeout=120_000)
        time.sleep(1.0)
        page.screenshot(path=str(args.out / "app_default.png"))

        # one click: load a never-seen test employee and predict
        page.get_by_role("button", name="Random test employee").click()
        page.wait_for_selector("#results table", timeout=120_000)
        time.sleep(1.5)
        page.screenshot(path=str(args.out / "app_random_employee.png"))

        # open the probabilities + sentence panels
        page.get_by_text("Probabilities", exact=True).click()
        page.get_by_text("What the model reads", exact=True).click()
        time.sleep(1.5)
        page.screenshot(path=str(args.out / "app_probabilities.png"), full_page=True)

        # edit a field and predict again (the "Actual" column disappears)
        page.get_by_role("tab", name="Job").click()
        time.sleep(0.8)
        overtime = page.locator("#in-OverTime input")
        other = "No" if overtime.input_value().strip() == "Yes" else "Yes"
        overtime.click()
        time.sleep(0.5)
        page.get_by_role("option", name=other, exact=True).click()
        time.sleep(0.5)
        page.get_by_role("button", name="Predict").click()
        page.wait_for_selector("text=Edited by hand")
        time.sleep(2.5)
        page.screenshot(path=str(args.out / "app_edited.png"))
        if errors:
            print("JavaScript errors:", errors)
        context.close()
        browser.close()

    if args.no_gif:
        return
    video = next(video_dir.glob("*.webm"), None)
    ffmpeg = ffmpeg_binary()
    if video and ffmpeg:
        gif = args.out / "demo.gif"
        filters = "fps=6,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-vf", filters, str(gif)], check=True)
        print(f"wrote {gif} ({gif.stat().st_size / 1e6:.1f} MB)")
    else:
        print("no video or no ffmpeg: GIF skipped")
    shutil.rmtree(video_dir, ignore_errors=True)
    print(f"screenshots in {args.out}")


if __name__ == "__main__":
    main()
