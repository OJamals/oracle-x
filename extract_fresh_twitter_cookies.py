import asyncio
import csv
import json
from playwright.async_api import async_playwright

PROXY = "http://spog4pgqhk:bgddgP0Wd8tFpd0~A6@gate.decodo.com:7000"
ACCOUNTS_FILE = "accounts_twscrape_checked.txt"
OUTPUT_FILE = "accounts_twscrape_fresh_cookies.txt"

async def login_and_get_cookies(playwright, username, password, user_agent=None):
    browser = await playwright.chromium.launch(proxy={"server": PROXY}, headless=True)
    context_args = {}
    if user_agent:
        context_args["user_agent"] = user_agent
    context = await browser.new_context(**context_args)
    page = await context.new_page()
    try:
        await page.goto("https://twitter.com/login", timeout=60000)
        await page.wait_for_selector('input[name="text"]', timeout=30000)
        await page.fill('input[name="text"]', username)
        await page.click('div[role="button"][data-testid="LoginForm_Login_Button"]', timeout=10000)
        await page.wait_for_timeout(2000)
        # Twitter may ask for username again
        if await page.query_selector('input[name="text"]'):
            await page.fill('input[name="text"]', username)
            await page.click('div[role="button"][data-testid="LoginForm_Login_Button"]', timeout=10000)
            await page.wait_for_timeout(2000)
        await page.wait_for_selector('input[name="password"]', timeout=30000)
        await page.fill('input[name="password"]', password)
        await page.click('div[role="button"][data-testid="LoginForm_Login_Button"]', timeout=10000)
        await page.wait_for_timeout(5000)
        # Wait for home page or check for login success
        if not await page.query_selector('nav[aria-label="Primary"]'):
            print(f"Login failed for {username}")
            await browser.close()
            return None
        cookies = await context.cookies()
        await browser.close()
        return cookies
    except Exception as e:
        print(f"Error logging in {username}: {e}")
        await browser.close()
        return None

async def main():
    accounts = []
    with open(ACCOUNTS_FILE, "r") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) < 2 or line.strip().startswith("#"):
                continue
            username = parts[0]
            password = parts[1]
            user_agent = None
            if len(parts) > 4 and parts[4] != "_":
                user_agent = parts[4]
            accounts.append((username, password, user_agent, line.strip()))

    async with async_playwright() as playwright:
        with open(OUTPUT_FILE, "w") as outf:
            for username, password, user_agent, original_line in accounts:
                print(f"Logging in: {username}")
                cookies = await login_and_get_cookies(playwright, username, password, user_agent)
                if cookies:
                    cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in cookies])
                    parts = original_line.split(":")
                    if len(parts) >= 6:
                        parts[5] = cookie_str
                    else:
                        parts.append(cookie_str)
                    outf.write(":".join(parts) + "\n")
                else:
                    print(f"Failed to get cookies for {username}")
    print("[INFO] Extraction complete. Output written to", OUTPUT_FILE)

if __name__ == "__main__":
    asyncio.run(main())
