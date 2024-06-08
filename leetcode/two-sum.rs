# https://leetcode.com/problems/two-sum/

use std::collections::HashMap;

impl Solution {
    pub fn two_sum0(nums: Vec<i32>, target: i32) -> Vec<i32> {
        for (i, n) in nums.iter().enumerate() {
            if let Some(j) = nums[..i].iter().position(
                    |&m| m + n == target) {
                return vec![i as i32, j as i32];
            }
        }
        unreachable!();
    }

    pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
        let mut hash_nums = HashMap::<i32, i16>::with_capacity(nums.len() / 2);
        for (i, n) in nums.into_iter().enumerate() {
            if let Some(i2) = hash_nums.get(&n) {
                return vec![*i2 as i32, i as i32];
            }
            hash_nums.insert(target - n, i as i16);
        }
        unreachable!();
    }
}

/*
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

 

Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
*/
