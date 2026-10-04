"""Helper FDSN Raspberry Shake con workaround cert expirado.
Docs oficiales usan wget --no-check-certificate; aqui ssl._create_unverified_context().
"""
import ssl
import time
import urllib.request
import urllib.error

BASE = "https://data.raspberryshake.org/fdsnws"
CTX = ssl._create_unverified_context()

def http_get(url, timeout=60, retries=3):
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, context=CTX, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")[:500] if hasattr(e, "read") else ""
            if e.code == 204:
                raise RuntimeError(f"204 No data: {url} {body}")
            if e.code == 429:
                time.sleep(5 * (i + 1))
                last = e
                continue
            raise RuntimeError(f"HTTP {e.code} {url}: {body}") from e
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"FAIL tras {retries}: {url}: {last}")

def dataselect(net, sta, loc, cha, start, end):
    url = (f"{BASE}/dataselect/1/query?net={net}&sta={sta}&loc={loc}&cha={cha}"
           f"&start={start}&end={end}")
    return http_get(url, timeout=90), url

def station_text(query):
    return http_get(f"{BASE}/station/1/query?{query}&format=text", timeout=30).decode()

def station_xml(net, sta, loc="00", cha="EHZ"):
    url = f"{BASE}/station/1/query?net={net}&sta={sta}&loc={loc}&cha={cha}&level=response"
    return http_get(url, timeout=30), url
