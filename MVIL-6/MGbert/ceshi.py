s = "ab#c"
t = "ad#c"
sl = list(s)
tl = list(t)
lens = len(sl)
lent = len(tl)
a_s = []
a_t = []
f = l = lens - 1
k = m = lent - 1
n = 0
num = 0
while f >= 0:
    while f >= 0 and sl[f] != "#":
        l = f
        f = f - 1
        if n == 0:
            a_s.append(sl[l])
        else:
            n = n - 1
    if f < 0:
        break
    if sl[f] == "#":
        n = n + 1
while k >= 0:
    while k >= 0 and sl[k] != "#":
        m = k
        k = k - 1
        if num == 0:
            a_t.append(tl[m])
        else:
            num = num - 1
    if f < 0:
        break
    if sl[k] == "#":
        num = num + 1
judge = 0
if len(a_s) == len(a_t):
    for i in range(len(a_t)):
        if a_t[i] == a_s[i]:
            judge = judge + 1
else:
    print(False)
if judge == len(a_s):
    print(True)