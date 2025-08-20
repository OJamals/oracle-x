import requests
from bs4 import BeautifulSoup

def test_finviz_market_breadth():
    """Test getting market breadth data from Finviz homepage"""
    print("Testing Finviz market breadth data extraction...")
    
    try:
        url = "https://finviz.com/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Look for market breadth indicators
        stats = {}
        for stat in soup.find_all('td', class_='snapshot-td2-cp'):
            text = stat.get_text()
            print(f"Found stat: {text}")
            if "Advancers" in text:
                stats["advancers"] = int(text.split()[1].replace(",", ""))
            if "Decliners" in text:
                stats["decliners"] = int(text.split()[1].replace(",", ""))
        
        print(f"Market breadth stats: {stats}")
        return stats
        
    except Exception as e:
        print(f"Error: {e}")
        return {}

if __name__ == "__main__":
    test_finviz_market_breadth()