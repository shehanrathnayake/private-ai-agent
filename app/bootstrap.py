import os

# Filesystem Contract - Paths relative to root
MEMORY_DIR = "memory"
SUMMARIES_DIR = os.path.join(MEMORY_DIR, "summaries")
MODELS_DIR = os.path.join(MEMORY_DIR, "models")
REPORTS_DIR = os.path.join(MEMORY_DIR, "reports")

KNOWLEDGE_FILE = os.path.join(MEMORY_DIR, "knowledge.md")
IDENTITY_FILE = os.path.join(MEMORY_DIR, "identity.md")
AGENT_DB = os.path.join(MEMORY_DIR, "agent_memory.db")
TOOL_AUDIT_LOG = os.path.join(MEMORY_DIR, "tool_audit.log")

def ensure_runtime_environment() -> None:
    """
    Bootstraps the application filesystem according to the contract.
    Ensures all required directories and default files exist.
    Safe to call multiple times (idempotent).
    """
    # 1. Ensure root memory directory exists
    os.makedirs(MEMORY_DIR, exist_ok=True)
    
    # 2. Define required sub-directories
    directories = [
        SUMMARIES_DIR,
        MODELS_DIR,
        REPORTS_DIR
    ]
    
    # 3. Migration logic: Move existing data from root back to memory/ if they exist in root
    migration_map = {
        "summaries": SUMMARIES_DIR,
        "models": MODELS_DIR,
        "reports": REPORTS_DIR,
        "knowledge.md": KNOWLEDGE_FILE,
        "identity.md": IDENTITY_FILE,
    }

    import shutil
    for old_path, new_path in migration_map.items():
        # Only migrate if the old root path exists and isn't already the new path
        if os.path.exists(old_path) and os.path.abspath(old_path) != os.path.abspath(new_path):
            try:
                if os.path.isdir(old_path):
                    os.makedirs(new_path, exist_ok=True)
                    for item in os.listdir(old_path):
                        s = os.path.join(old_path, item)
                        d = os.path.join(new_path, item)
                        if not os.path.exists(d):
                            shutil.move(s, d)
                    # Try to remove old dir if empty
                    try: os.rmdir(old_path)
                    except: pass
                else:
                    # Move file if the destination doesn't exist
                    if not os.path.exists(new_path):
                        shutil.move(old_path, new_path)
                    else:
                        # Destination already exists, root version might be redundant
                        # For safety, we only remove if they are identical or root is default
                        pass
                # print(f"[BOOTSTRAP] Migrated {old_path} to {new_path}")
            except Exception as e:
                print(f"[BOOTSTRAP] Migration warning for {old_path}: {e}")

    # 4. Create required directories (if they don't exist after migration)
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create required directory {directory}: {e}")

    # 5. Setup Required-at-runtime files
    if not os.path.exists(KNOWLEDGE_FILE):
        try:
            with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
                f.write("# Knowledge Base\n")
        except Exception as e:
            raise RuntimeError(f"Failed to create required file {KNOWLEDGE_FILE}: {e}")

    # 6. Ensure sub-directories
    os.makedirs(os.path.join(MEMORY_DIR, "vector_index"), exist_ok=True)
