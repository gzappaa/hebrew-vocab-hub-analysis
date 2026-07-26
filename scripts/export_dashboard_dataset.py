import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import os

os.makedirs("data", exist_ok=True)






load_dotenv()

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

sns.set_theme(style="whitegrid")



query = """
SELECT 
    ws.word_id,
    w.word,
    ws.songs,
    ws.news,
    ws.youtube,
    ws.total
FROM word_sources ws
JOIN words w
ON ws.word_id = w.id;
"""

df_sources = pd.read_sql(query, engine)

df_sources["source_count"] = (
    (df_sources["songs"] > 0).astype(int) +
    (df_sources["news"] > 0).astype(int) +
    (df_sources["youtube"] > 0).astype(int)
)


word_ids_str = "', '".join(
    df_sources["word_id"].dropna().astype(str)
)

query = f"""
SELECT 
    wl.word_id,
    l.part_of_speech
FROM word_lemmas wl
JOIN lemmas l
ON l.id = wl.lemma_id
WHERE wl.word_id IN ('{word_ids_str}');
"""

df_pos = pd.read_sql(query, engine)

df_pos = (
    df_pos
    .groupby("word_id")["part_of_speech"]
    .apply(lambda x: ", ".join(sorted(set(x.dropna()))))
    .reset_index()
)

df_sources = df_sources.merge(
    df_pos,
    on="word_id",
    how="left"
)


df_sources.to_csv(
    "data/hebrew_vocab_dashboard.csv",
    index=False,
    encoding="utf-8-sig"
)


