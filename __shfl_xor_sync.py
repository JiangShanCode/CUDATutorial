from copy import deepcopy

ans_pre = [[i] for i in range(32)]

laneMask = 16
while(laneMask > 0):
    cur_ans = deepcopy(ans_pre)
    for i in range(32):
        for num in ans_pre[i ^ int(laneMask)]:
            cur_ans[i].append(num)
    
    print(cur_ans[0])
    laneMask = laneMask >> 1
    ans_pre = cur_ans