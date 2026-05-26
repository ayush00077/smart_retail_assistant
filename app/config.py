from dotenv import load_dotenv
import os
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

DATABASE_NAME = os.getenv("DATABASE_NAME")

# from dotenv import load_dotenv
# import os
# from pathlib import Path

# # Get backend directory
# BASE_DIR = Path(__file__).resolve().parent.parent

# # Load .env file
# load_dotenv(BASE_DIR / ".env")

# MONGO_URL = os.getenv("MONGO_URL")
# DATABASE_NAME = os.getenv("DATABASE_NAME")

# print("Mongo URL:", MONGO_URL)

# from dotenv import load_dotenv
# import os

# load_dotenv(".env")

# MONGO_URL = os.getenv("MONGO_URI")
# DATABASE_NAME = os.getenv("DATABASE_NAME")

# print("Mongo URI:", MONGO_URI)


# from dotenv import load_dotenv
# import os
# from pathlib import Path

# # Get backend root directory
# BASE_DIR = Path(__file__).resolve().parent.parent.parent

# # Load .env file
# load_dotenv(BASE_DIR / ".env")

# MONGO_URL = os.getenv("MONGO_URL")
# DATABASE_NAME = os.getenv("DATABASE_NAME")

# print("Mongo URL:", MONGO_URL)