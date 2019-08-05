# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

class Solution(object):
    def twoSum(nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        n = len(nums)
        for i in range(n):
            for j in range(i+1,n):
                sum = nums[i]+nums[j]
                if sum==target:
                    return [i,j]
        dic = dict(zip(nums,range(n)))
        for i in range(n):
            sub = target-nums[i]
            
            if sub in dic and dic[sub]!=i:
                return [i,dic[sub]]
        return  0

# import ptvsd
# ptvsd.enable_attach()
# ptvsd.wait_for_attach()

import csv
with open('mrcnn/filelist.csv') as f:
    filelist = csv.reader(f)
    
    for obj in filelist:
        print(obj)
        break
# print(filelist[0])

nums = [2, 7, 11, 15]
target = 9
n = len(nums)
dic = dict(zip(nums,range(n)))
ans = Solution.twoSum(nums, target)
