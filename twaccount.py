
import asyncio
from twscrape.accounts_pool import AccountsPool

PROXY = "http://spog4pgqhk:bgddgP0Wd8tFpd0~A6@gate.decodo.com:7000"

async def main():
    pool = AccountsPool()
    # Open your formatted accounts file
    with open("accounts_twscrape_checked.txt", "r") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) < 6:
                continue  # skip malformed lines

            username = parts[0]
            password = parts[1]
            email = parts[2]
            email_password = parts[3]
            user_agent = None if parts[4] == "_" else parts[4]
            cookies = parts[5].lstrip("_") if parts[5].startswith("_") else parts[5]

            print(f"Adding {username}...")

            await pool.add_account(
                username=username,
                password=password,
                email=email,
                email_password=email_password,
                user_agent=user_agent,
                proxy=PROXY,
                cookies=cookies
            )

    await pool.save()
    print("All accounts added and saved!")

if __name__ == "__main__":
    asyncio.run(main())