"""Minimal RawTree HTTP client used by the simulator and the admin script."""
import gzip
import json
import os
import time

import requests


class RawTree:
    def __init__(self, key=None, url=None):
        self.url = (url or os.environ.get("RAWTREE_API_URL", "https://api.rawtree.com")).rstrip("/")
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {key or os.environ['RAWTREE_API_KEY']}"

    def _call(self, method, path, retries=5, **kw):
        for attempt in range(retries):
            try:
                r = self.s.request(method, self.url + path, timeout=kw.pop("timeout", 60), **kw)
            except requests.RequestException:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 502, 503, 504) and attempt < retries - 1:
                time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
                continue
            if r.status_code == 400 and "Database not found" in r.text and attempt < retries - 1:
                time.sleep(0.2 * (attempt + 1))  # a replica that just joined the cluster is still creating databases: retry on another
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:800]}")
            return r.json() if r.content else None

    def query(self, sql, database=None, params=None):
        body = {"sql": sql, **({"params": params} if params else {})}
        return self._call("POST", "/v1/query", params={"database": database} if database else None, json=body)["data"]

    def execute(self, sql, database=None):
        return self._call("POST", "/v1/query", params={"database": database} if database else None, json={"sql": sql})

    def insert(self, database, table, rows, gz=False):
        if not rows:
            return None
        body = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
        headers = {"Content-Type": "application/json"}
        if gz:
            body, headers["Content-Encoding"] = gzip.compress(body, 3), "gzip"
        return self._call("POST", f"/v1/tables/{table}", params={"database": database}, data=body, headers=headers, timeout=600)

    def databases(self):
        return [d["name"] if isinstance(d, dict) else d for d in self._call("GET", "/v1/databases")["databases"]]

    def create_database(self, name):
        return self._call("POST", "/v1/databases", json={"name": name})

    def create_table(self, database, name, sorting_key=None):
        return self._call("POST", "/v1/tables", params={"database": database}, json={"name": name, "sorting_key": sorting_key})

    def drop_table(self, database, name):
        return self._call("DELETE", f"/v1/tables/{name}", params={"database": database})

    def tables(self, database):
        return [t["name"] if isinstance(t, dict) else t for t in self._call("GET", "/v1/tables", params={"database": database})["tables"]]

    def create_key(self, name, roles, database):
        return self._call("POST", "/v1/keys", params={"database": database}, json={"name": name, "database_roles": roles})
