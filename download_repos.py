"""download_repos.py — 通过 GitHub API 下载 AwareLiquid 组织的仓库 (绕过被墙的 git clone)。"""
import subprocess
import urllib.request
import urllib.error
import tarfile
import io
import os

TOKEN = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()
OUT = "E:/awareliquid_repos"
os.makedirs(OUT, exist_ok=True)

REPOS = ["AwareLiquid-Web"]


def default_branch(repo):
    req = urllib.request.Request(
        f"https://api.github.com/repos/AwareLiquid/{repo}",
        headers={"Authorization": f"Bearer {TOKEN}", "User-Agent": "curl"})
    with urllib.request.urlopen(req, timeout=30) as r:
        import json
        return json.loads(r.read()).get("default_branch", "main")


def download(repo):
    branch = default_branch(repo)
    for br in [branch, "main", "master"]:
        url = f"https://api.github.com/repos/AwareLiquid/{repo}/tarball/{br}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}",
                                                   "User-Agent": "curl"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            tf = tarfile.open(fileobj=io.BytesIO(data))
            tf.extractall(OUT)
            # 打印文件数
            names = tf.getnames()
            print(f"  OK {repo} (branch={br}): {len(names)} 文件")
            return
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            print(f"  FAIL {repo}: HTTP {e.code}")
            return
        except Exception as e:
            print(f"  FAIL {repo}: {e}")
            return
    print(f"  FAIL {repo}: 所有分支都 404")


print(f"下载到 {OUT}/")
for r in REPOS:
    download(r)

print("\n=== 目录结构 ===")
for r in REPOS:
    d = os.path.join(OUT, r)
    print(f"\n[{r}]")
    if os.path.isdir(d):
        for item in sorted(os.listdir(d))[:30]:
            print(f"  {item}")
    else:
        # tarball 解压后目录名带 commit hash 前缀, 找一下
        for sub in os.listdir(OUT):
            if sub.startswith("AwareLiquid-" + r) or r.lower() in sub.lower():
                print(f"  解压目录: {sub}")
                for item in sorted(os.listdir(os.path.join(OUT, sub)))[:30]:
                    print(f"    {item}")
