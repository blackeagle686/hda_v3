import subprocess
import sys
import os
from datetime import datetime

def git_push(commit_msg=None):
    if not commit_msg:
        commit_msg = f"Auto-commit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    try:
        # 1️⃣ git add .
        subprocess.run(["git", "add", "."], check=True)
        
        # 2️⃣ git commit -m "message"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        
        # 3️⃣ git push
        subprocess.run(["git", "push", "origin", "master"], check=True)
        
        print("[✅] Changes pushed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"[❌] Git command failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    msg = None
    if len(sys.argv) > 1:
        msg = " ".join(sys.argv[1:])
    git_push(commit_msg=msg)
