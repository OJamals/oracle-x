import requests
from bs4 import BeautifulSoup
import re


def extract_market_breadth():
    """Extract market breadth data from Finviz homepage"""
    try:
        url = "https://finviz.com/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Look for the market breadth section in the text
        page_text = soup.get_text()

        # Extract advancing/declining data using regex
        # Look for patterns like "Advancing47.4% (4885) Declining(5007) 48.5%"
        advancing_pattern = r"Advancing[\d.]+%\s*\((\d+)\)"
        declining_pattern = r"Declining\((\d+)\)"

        advancing_match = re.search(advancing_pattern, page_text)
        declining_match = re.search(declining_pattern, page_text)

        advancers = int(advancing_match.group(1)) if advancing_match else None
        decliners = int(declining_match.group(1)) if declining_match else None

        print(f"Advancers: {advancers}")
        print(f"Decliners: {decliners}")

        # Also look for new highs/lows
        new_high_pattern = r"New High[\d.]+%\s*\((\d+)\)"
        new_low_pattern = r"New Low\((\d+)\)"

        high_match = re.search(new_high_pattern, page_text)
        low_match = re.search(new_low_pattern, page_text)

        new_highs = int(high_match.group(1)) if high_match else None
        new_lows = int(low_match.group(1)) if low_match else None

        print(f"New Highs: {new_highs}")
        print(f"New Lows: {new_lows}")

        return {
            "advancers": advancers,
            "decliners": decliners,
            "new_highs": new_highs,
            "new_lows": new_lows,
        }

    except Exception as e:
        print(f"Error extracting market breadth: {e}")
        return {}


if __name__ == "__main__":
    print("Extracting market breadth data from Finviz homepage...")
    result = extract_market_breadth()
    print(f"Result: {result}")
