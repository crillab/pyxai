import user as u


# Generalize
print("check generalize")
assert(u.generalize([1], [2, 1]))
assert(u.generalize([1], [1, -2]))
assert(u.generalize([1, -2], [1, -2]))
assert(u.generalize([1, -2], [2, 4, 5]) is False)
assert(u.generalize([1], [2, 4, 5]) is False)


# Check conflict
print("check conflict")
# Conflict
rule1 = [2, 3]
rule2 = [4, 2]
assert(u.conflict(rule1, rule2))
assert(u.conflict(rule2, rule1))
assert(u.conflict([-2], [3]))
assert(u.conflict([2], [-2]) is False)
assert(u.conflict([-2], [2]) is False)



# Use cases
rules = [[2, 3], [4, 2]]
rule = [4, 2]
rule_AI = [1, 9, 0]

new_rule_AI = u.replace(rules, rule, rule_AI)
print("Règle AI après remplacement :", new_rule_AI)
