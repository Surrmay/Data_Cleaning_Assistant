import os

# Define your folder structure
structure = {
    "app.py": "Copy from \"app.py - Streamlit User Interface\"",
    "inspector.py": "Copy from \"inspector.py - Data Inspection Engine\"",
    "qa_service.py": "Copy from \"qa_service.py - LLM Integration Layer\"",
    "utils.py": "Copy from \"utils.py - Helper Functions\"",
    "requirements.txt": "Copy from \"requirements.txt\"",
    ".env.example": "Copy from \".env.example\"",
    ".gitignore": "Copy from \".gitignore\"",
    "README.md": "Copy from \"README.md\"",
    "data/sample.csv": "Copy from \"sample.csv\"",
    "tests/test_inspector.py": "Copy from \"test_inspector.py\"",
}

def setup_structure(base_path="."):
    for path, desc in structure.items():
        full_path = os.path.join(base_path, path)
        folder = os.path.dirname(full_path)

        # Create directories if they don't exist
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            print(f"📁 Created directory: {folder}")

        # Create placeholder files with description comment
        if not os.path.exists(full_path):
            with open(full_path, "w", encoding="utf-8") as f:
                if full_path.endswith((".py", ".txt", ".md", ".gitignore", ".example", ".csv")):
                    f.write(f"# {desc}\n")
                else:
                    f.write(f"{desc}\n")
            print(f"📝 Created file: {full_path}")
        else:
            print(f"✅ Already exists: {full_path}")

if __name__ == "__main__":
    print("🚀 Setting up project structure...\n")
    setup_structure()
    print("\n✅ Project structure setup complete!")
