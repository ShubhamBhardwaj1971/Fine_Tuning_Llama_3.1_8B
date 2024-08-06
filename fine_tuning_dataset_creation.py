import psycopg2
import json
from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()

# Database connection setup using environment variables
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

def connect_db():
    """Connect to the PostgreSQL database."""
    try:
        return psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
    except psycopg2.Error as e:
        raise RuntimeError(f"Database connection error: {e}")

# Function to extract and process research paper details
def extract_paper_details():
    conn = connect_db()
    cur = conn.cursor()

    # Example query to fetch paper details
    query = """
    SELECT rp.title, a.name, rp.keywords, rp.abstract, rp.conflict_of_interest, COUNT(c.id) AS citation_count
    FROM research_papers rp
    LEFT JOIN authors a ON rp.id = a.paper_id
    LEFT JOIN citations c ON rp.id = c.citing_paper_id
    GROUP BY rp.id, a.name;
    """
    cur.execute(query)
    rows = cur.fetchall()

    # Generate dataset for fine-tuning
    dataset = []

    for row in rows:
        title, author, keywords, abstract, conflict_of_interest, citation_count = row
        
        # Create entry for each question type
        dataset.append({
            "instruction": f"Who are the authors of the paper titled '{title}'?",
            "input": None,
            "output": f"The authors of the paper '{title}' include {author}."
        })
        
        dataset.append({
            "instruction": f"Keywords",
            "input": f"{title}",
            "output": f"The keywords for this paper are {keywords}."
        })
        
        dataset.append({
            "instruction": f"Has the paper titled '{title}' been cited?",
            "input": None,
            "output": f"Yes, the paper '{title}' has been cited by {citation_count} other papers." if citation_count > 0 else f"No, the paper '{title}' has not been cited yet."
        })
        
        dataset.append({
            "instruction": f"Does the paper '{title}' have any declared conflicts of interest?",
            "input": None,
            "output": f"{conflict_of_interest}" if conflict_of_interest else f"No, the paper '{title}' has no declared conflicts of interest."
        })
        
        dataset.append({
            "instruction": f"Abstract",
            "input": f"{title}",
            "output": abstract
        })

    # Save dataset to a JSONL file
    with open("fine_tuning_dataset.jsonl", "w") as f:
        for entry in dataset:
            json.dump(entry, f)
            f.write("\n")

    cur.close()
    conn.close()

# Run the data extraction process
if __name__ == "__main__":
    extract_paper_details()
