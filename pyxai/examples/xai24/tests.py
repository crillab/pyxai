import user as u
import coverage
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




rule = (-12, 16, -22, 30, 41, -99, -127, 199, 218, 226, -273, -296, -345)
ia = (2, 16, 21, 31, -37, 112, 406)
print("ici ", u.conflict(rule, ia))

print("coverage")
sigma = [[1, -2], [-3, -4]]
nb_v = 5
positives = [[2, 4], [1, 5]]
negatives = [[-1, -4]]

cvg = coverage.Coverage(sigma, nb_v , None, None)
cvg.test(sigma, positives, negatives, nb_v)
