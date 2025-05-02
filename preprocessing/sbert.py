from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L12-v2")

ref_sentences = [
    "flu medicine",
    "flu vaccine children",
    "flu virus symptoms",
    "best medicine for flu",
    "influenza treatment",
    """Flu symptoms come on very quickly and can include: a sudden high temperature,
    an aching body, feeling tired or exhausted, a dry cough, a sore throat,
    a headache, difficulty sleeping, loss of appetite, diarrhoea or tummy pain,
    feeling sick and being sick. The symptoms are similar for children,
    but they can also get pain in their ear and appear less active.""",
    """The flu, or influenza, is a contagious respiratory illness caused by
    influenza viruses. It affects the nose, throat, and sometimes the lungs,
    leading to symptoms like fever, chills, muscle aches, cough, sore throat,
    runny nose, and fatigue. The flu spreads through respiratory droplets when an
    infected person coughs, sneezes, or talks. While most people recover within a
    few days to two weeks, it can cause severe complications like pneumonia,
    especially in young children, the elderly, or individuals with weakened
    immune systems. Annual flu vaccinations help prevent infection and reduce
    the severity of symptoms if contracted.""",
]

ref_embeddings = model.encode(ref_sentences)

queries_path = "../processed_data/filtered_queries.csv"
scores = []
with open(queries_path, "r", encoding="utf-8", errors="replace") as queries_file:
    print("embedding...")
    idx = 1
    while True:
        query = queries_file.readline()[:-1]
        if not query:
            break
        embedding = model.encode(query).reshape(1, -1)
        similarities = cosine_similarity(embedding, ref_embeddings)
        score = sum(similarities[0]) / len(ref_embeddings)
        scores.append((idx, score))
        idx += 1

print("sorting...")
scores.sort(key=lambda x: x[1], reverse=True)

qfile = open(queries_path, encoding="utf-8", errors="replace")

queries = qfile.readlines()

print("writing...")
with open("../processed_data/scores.csv", "w", encoding="utf-8") as scores_file:
    scores_file.write("index,score,query\n")
    for s in scores:
        scores_file.write(f"{s[0]},{s[1]},{queries[s[0]-1]}")

qfile.close()
