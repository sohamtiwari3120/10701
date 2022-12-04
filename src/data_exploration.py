import os
import json
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def read_patent_jsonl(path: str, use_tqdm: bool=False) -> List[dict]:
    """list of dicts data to be loaded"""
    data: List[dict] = []
    with open(path) as f:
        for line in tqdm(f, disable=not(use_tqdm)):
            data.append(json.loads(line.strip()))
    
    return data

DATA_PATH = "patent_acceptance_pred_subs=0.2_no_desc.jsonl"
data = read_patent_jsonl(DATA_PATH, use_tqdm=True) # load list of JSONs.
df = pd.DataFrame(data) # create data frame from list of JSONs.
topic_dist = df["topic_id"].apply(lambda x: x[:4]).value_counts()
top15_cpc_topics = ['G06F', 'H01L', 'H04L', 'A61K', 'H04W', 'A61B', 'G06Q', 'H04N', 'G01N', 'G02B', 'H01M', 'A61F', 'G06T', 'C07D']
top15_cpc_topic_names = ["ELECTRIC DIGITAL\nDATA PROCESSING", "SEMICONDUCTOR\nDEVICES", "TRANSMISSION OF\nDIGITAL INFORMATION", "PREPARATIONS FOR\nMEDICAL PURPOSES", "WIRELESS\nCOMMUNICATION", "DIAGNOSIS\nSURGERY", "DATA PROCESSING\nSYSTEMS", "TELEVISION", "CHEMICAL\nOR PHYSICAL\nANALYSIS", "OPTICAL SYSTEMS", "CHEMICAL\nENERGY TO\nELECTRICAL ENERGY", "BIO IMPLANTS", "IMAGE DATA\nPROCESSING", "HETEROCYCLIC\nCOMPOUNDS"]
top15_cpc_topic_names = [t.title() for t in top15_cpc_topic_names]
# stats:
# 1. we have 764054 patents (477980 have blank topic label).
# 2. 594 main topics (based on first 4 characters of main cpc label)
# 3. top 2-15 topics have 124717 instances.
# 4. remaining 16-594 have 161357 instances.

os.makedirs("plots", exist_ok=True)
# topic distribution plot for top 2-15 topics.
plt.clf()
plt.figure(dpi=300)
x = list(range(15-1))
xlbls = topic_dist.nlargest(15)[1:].keys().tolist()
y = topic_dist.nlargest(15)[1:].tolist()
plt.bar(x, y)
plt.xticks(x, top15_cpc_topic_names, 
rotation=90, fontsize=8)
plt.xlabel("Topics")
plt.ylabel("No. of instances")
plt.title("No. of instances for the top 15 topics")
plt.tight_layout()
plt.savefig("plots/top_15_topics.png")

# topic rank-frequency plot (Zipf like plot)
plt.clf()
plt.figure(dpi=300)
W = 0.2
ranks = [(2, 5), (6, 10), (11, 25), (26, 50), (51, 100), (101, 250), (251, 500), (500, 594)]
rank_labels = [f"{i} - {j}" for i,j in ranks]
y = [topic_dist.nlargest(j).sum()-topic_dist.nlargest(i-1).sum() for i, j in ranks[:-1]]+[topic_dist.sum()-topic_dist.nlargest(500).sum()]
plt.bar(range(len(ranks)), y, 
color="orange", width=W, alpha=0.7)
env_x, env_y = [], []
for i in range(len(ranks)):
    env_x.append(i-W/2)
    env_x.append(i+W/2)
    env_y.append(y[i])
    env_y.append(y[i])
plt.plot(env_x, env_y, 
color="orange", alpha=0.7)
plt.fill_between(
x=env_x, y1=env_y, 
color="orange", alpha=0.7) # where= (-1 < t)&(t < 1)
plt.xlabel("topic rank range")
plt.ylabel("No. of instances")
plt.xticks(range(len(ranks)), rank_labels, rotation=45)
plt.title("No. of instances for range of topic ranks")
plt.tight_layout()
plt.savefig("plots/topic_rank_freq_dist.png")

# explore the inventors.

# the distribution of no. of inventors per patent.
num_inventor_dist = df['inventor_list'].apply(lambda x: len(x)).value_counts()
# collapse the last one into 10 or more inventors.
num_inventor_dist[10] += num_inventor_dist[10:].sum()
num_inventor_dist = num_inventor_dist[:10]

# the distribution of no. of inventors per patent.
num_inventor_dist = df['inventor_list'].apply(lambda x: len(x)).value_counts()
# collapse the last one into 10 or more inventors.
num_inventor_dist[10] += num_inventor_dist[10:].sum()
num_inventor_dist = num_inventor_dist[:10]

# histogram of number of no. of inventors.
plt.clf()
plt.figure(dpi=300)
x = list(range(10))
W = 0.2
xlbls = [f"{i+1}" for i in range(9)]+["10 or more"]
y = num_inventor_dist.tolist()
# plt.bar(x, y, color="red", width=0.4)
plt.bar(x, y, color="red", width=W)
env_x, env_y = [], []
for i in range(10):
    env_x.append(i-W/2)
    env_x.append(i+W/2)
    env_y.append(y[i])
    env_y.append(y[i])
plt.fill_between(env_x, env_y, color="red", alpha=0.6) 
# where= (-1 < t)&(t < 1)
plt.xticks(x, xlbls)
plt.xlabel("No. of inventors")
plt.ylabel("No. of instances")
plt.title("No. of patents with x inventors")
plt.tight_layout()
plt.savefig("plots/num_inventors_dist.png")

# no of applications assesed by top 15 examiners.
plt.clf()
plt.figure(dpi=300)
x = list(range(15))
y = df["examiner_namer"].apply(lambda x: ' '.join(x.split())
).value_counts()[1:].nlargest(15)
xlbls = df["examiner_namer"].apply(lambda x: ' '.join(x.split(
)).title()).value_counts()[1:].nlargest(15).keys().tolist()
plt.bar(x, y, color="green")
for i, v in enumerate(y):
    plt.text(x=i-0.4, y=v+10, 
    s=f"{v}", fontsize=10)
plt.xticks(x, xlbls, rotation=90)
plt.xlabel("Name of examiner")
plt.ylabel("No. of applications examined")
plt.title("No. of applications examined for top 15 examiners")
plt.tight_layout()
plt.savefig("plots/examiner_dist.png")

# rejection rate of top 15 examiners.
top15_examiners = df["examiner_namer"].value_counts(
)[1:].nlargest(15).keys().tolist()
plt.clf()
plt.figure(dpi=300)
x = list(range(15))
def get_reject_rate(df: pd.DataFrame) -> float:
    N1 = len(df[df["decision"] == "ACCEPTED"])
    N2 = len(df[df["decision"] == "REJECTED"])
    return round(100*N2/(N1+N2), 2)
y = [get_reject_rate(df[df["examiner_namer"] == name]) for name in top15_examiners]
xlbls = df["examiner_namer"].apply(lambda x: ' '.join(x.split(
)).title()).value_counts()[1:].nlargest(15).keys().tolist()
plt.bar(x, y, color="#6632a8")
for i, v in enumerate(y):
    plt.text(x=i-0.5, y=v+1, 
    s=f"{v}", fontsize=8)
plt.xticks(x, xlbls, rotation=90)
plt.xlabel("Name of examiner")
plt.ylabel("Rejection rate of examiner")
plt.title("Rejection rate of the top-15 patent examiners")
plt.tight_layout()
plt.savefig("plots/examiner_accept_rate.png")