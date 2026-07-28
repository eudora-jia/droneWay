#!/usr/bin/env python3
"""Download publicly linked MP4 files from flyability.com, respectfully."""
import argparse
import hashlib
import os
import re
import time
import urllib.parse
import urllib.robotparser
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

BASE_URL = "https://www.flyability.com/"
USER_AGENT = "FlyabilityPublicVideoArchiver/1.0 (personal offline viewing)"
TIMEOUT = 30
DELAY_SECONDS = 1.0


def same_site(url):
    host = urllib.parse.urlparse(url).hostname or ""
    return host == "flyability.com" or host.endswith(".flyability.com")


def absolute(url, base):
    return urllib.parse.urldefrag(urllib.parse.urljoin(base, url))[0]


def allowed(robots, url):
    return same_site(url) and robots.can_fetch(USER_AGENT, url)


def fetch(session, robots, url):
    if not allowed(robots, url):
        return None
    time.sleep(DELAY_SECONDS)
    try:
        response = session.get(url, timeout=TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        return response
    except requests.RequestException as error:
        print(f"Skip {url}: {error}")
        return None


def sitemap_urls(session, robots):
    seen, pending, pages = set(), [absolute("/sitemap.xml", BASE_URL)], set()
    while pending:
        url = pending.pop()
        if url in seen or not allowed(robots, url):
            continue
        seen.add(url)
        response = fetch(session, robots, url)
        if response is None:
            continue
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError:
            continue
        for node in root.iter():
            if node.tag.endswith("loc") and node.text:
                target = node.text.strip()
                if target.endswith(".xml"):
                    pending.append(target)
                elif same_site(target):
                    pages.add(target)
    return pages


def discover_mp4_urls(html, page_url):
    candidates = re.findall(r"(?:https?:)?//[^\s<>]+?\.mp4(?:\?[^\s<>]*)?|[^\s<>]+?\.mp4(?:\?[^\s<>]*)?", html, re.I)
    return {absolute(url, page_url) for url in candidates}


def save_video(session, url, output_dir):
    parsed = urllib.parse.urlparse(url)
    original = os.path.basename(parsed.path) or "video.mp4"
    stem, extension = os.path.splitext(original)
    digest = hashlib.sha256(url.encode()).hexdigest()[:10]
    target = output_dir / f"{stem}-{digest}{extension or '.mp4'}"
    if target.exists():
        print(f"Exists: {target.name}")
        return
    try:
        with session.get(url, stream=True, timeout=TIMEOUT) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "video" not in content_type and not parsed.path.lower().endswith(".mp4"):
                print(f"Skip non-video: {url}")
                return
            with target.open("wb") as handle:
                for chunk in response.iter_content(1024 * 256):
                    if chunk:
                        handle.write(chunk)
        print(f"Saved: {target}")
    except requests.RequestException as error:
        target.unlink(missing_ok=True)
        print(f"Download failed {url}: {error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="flyability_videos", help="Download directory")
    parser.add_argument("--max-pages", type=int, default=1000, help="Maximum public pages to scan")
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    robots = urllib.robotparser.RobotFileParser()
    robots.set_url(absolute("/robots.txt", BASE_URL))
    robots.read()
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    pages = sorted(sitemap_urls(session, robots))[:args.max_pages]
    print(f"Scanning {len(pages)} public pages")
    videos = set()
    for index, page in enumerate(pages, 1):
        print(f"[{index}/{len(pages)}] {page}")
        response = fetch(session, robots, page)
        if response and "text/html" in response.headers.get("content-type", ""):
            videos.update(discover_mp4_urls(response.text, response.url))
    print(f"Found {len(videos)} MP4 URLs")
    for url in sorted(videos):
        save_video(session, url, output_dir)


if __name__ == "__main__":
    main()
