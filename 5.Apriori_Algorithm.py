transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Diapers', 'Beer', 'Eggs'],
    ['Milk', 'Diapers', 'Beer', 'Cola'],
    ['Bread', 'Milk', 'Diapers', 'Beer'],
    ['Bread', 'Milk', 'Diapers', 'Cola']
]

# Generate itemset-1
def itemset_1(transactions):
    itemsets = set()
    for T in transactions:
        for item in T:
            itemsets.add(frozenset([item]))
    return itemsets


# Support function (works for any size itemset)
def get_support(itemset, transactions):
    count = 0
    for transaction in transactions:
        if itemset.issubset(set(transaction)):
            count += 1
    return count / len(transactions)


# Apriori Algorithm
def apriori(transactions, min_support):

    itemsets = itemset_1(transactions)
    support_data = {}
    L1 = []

    print("\nL1 (Frequent 1-itemsets)")
    for item in itemsets:
        support = get_support(item, transactions)
        if support >= min_support:
            L1.append(item)
            support_data[item] = support
            print(set(item), "Support:", round(support, 2))

    L = [L1]
    k = 2

    while len(L[k-2]) > 0:
        candidates = []

        for i in range(len(L[k-2])):
            for j in range(i+1, len(L[k-2])):
                union_set = L[k-2][i] | L[k-2][j]
                if len(union_set) == k:
                    candidates.append(union_set)

        candidates = list(set(candidates))
        Lk = []

        print(f"\nL{k} (Frequent {k}-itemsets)")

        for candidate in candidates:
            support = get_support(candidate, transactions)
            if support >= min_support:
                Lk.append(candidate)
                support_data[candidate] = support
                print(set(candidate), "Support:", round(support, 2))

        if not Lk:
            break

        L.append(Lk)
        k += 1

    return L, support_data


# Generate Strong Rules
def generate_rules(frequent_itemsets, support_data, min_confidence):

    print("\nStrong Association Rules")
    print("-------------------------------------")

    for level in frequent_itemsets[1:]:
        for itemset in level:
            for item in itemset:

                antecedent = frozenset([item])
                consequent = itemset - antecedent

                confidence = support_data[itemset] / support_data[antecedent]
                lift = confidence / support_data[consequent]

                if confidence >= min_confidence and lift > 1:
                    print(f"{set(antecedent)} → {set(consequent)}")
                    print("Support:", round(support_data[itemset],2),
                          "Confidence:", round(confidence,2),
                          "Lift:", round(lift,2)) 
                                
# determining minimum support and confidence
min_support = 0.4
min_confidence = 0.6


frequent_itemsets, support_data = apriori(transactions, min_support)
generate_rules(frequent_itemsets, support_data, min_confidence)
