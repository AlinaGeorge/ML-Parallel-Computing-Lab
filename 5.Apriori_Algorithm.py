from itertools import combinations

# Dataset
transactions = [
    ['Milk','Bread','Butter'],
    ['Bread','Butter'],
    ['Milk','Bread'],
    ['Milk','Butter'],
    ['Bread','Butter'],
    ['Milk','Bread','Butter']
]

# Support function
def support(itemset):
    return sum(set(itemset).issubset(t) for t in transactions) / len(transactions)

# Get all items
items = set(i for t in transactions for i in t)

min_sup = 0.5
min_conf = 0.7

# ----------------------------
# Frequent Itemsets
# ----------------------------
freq = {}

for k in range(1, len(items)+1):
    for comb in combinations(items, k):
        s = support(comb)
        if s >= min_sup:
            freq[frozenset(comb)] = s

print("Frequent Itemsets:")
for i, s in freq.items():
    print(set(i), ":", round(s,2))

# ----------------------------
# Association Rules
# ----------------------------
print("\nRules:")
for itemset in freq:
    if len(itemset) > 1:
        for i in range(1, len(itemset)):
            for a in combinations(itemset, i):
                a = frozenset(a)
                c = itemset - a

                conf = freq[itemset] / freq[a]
                if conf >= min_conf:
                    lift = conf / freq[c]
                    print(set(a), "→", set(c),
                          "| Conf:", round(conf,2),
                          "Lift:", round(lift,2))
