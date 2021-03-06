"""
@graduation project
@School of Computer Science and Engineering, Beihang University
@author  : hyc
@File    : val.py
@contact : hyc2026@yeah.net
"""
def normal_leven(str1, str2):
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1
    # create matrix
    matrix = [0 for n in range(len_str1 * len_str2)]
    # init x axis
    for i in range(len_str1):
        matrix[i] = i
    # init y axis
    for j in range(0, len(matrix), len_str1):
        if j % len_str1 == 0:
            matrix[j] = j // len_str1

    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[j * len_str1 + i] = min(matrix[(j - 1) * len_str1 + i] + 1,
                                           matrix[j * len_str1 + (i - 1)] + 1,
                                           matrix[(j - 1) * len_str1 + (i - 1)] + cost)

    return matrix[-1]

gt_list = {}
n = 0
for line in open("gt.txt"):
    if len(line) > 3:
        l = line.split(' ')
        gt_list[l[0][0:-1]] = l[1][1:-2].lower()
i = 0.0
for line in open("res_none.txt"):
    if len(line) > 3:
        n += 1
        l = line.split(' ')
        if gt_list[l[0]] == l[1][0:-1].lower():
            i = i + 1
        else:
            # s = normal_leven(gt_list[l[0]], l[1][0:-1].lower())
            # i = i + 1 - s / max(len(gt_list[l[0]]), len(l[1][0:-1].lower()))
            print(l[0], gt_list[l[0]], l[1][0:-1].lower())
print(i/n)

