## 1.二维数组中的查找 ##

在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

思路：

从[0][n-1]处开始查找。

	public class Solution {
	    public boolean Find(int target, int [][] array) {
	        if(array == null || array.length == 0 || array[0].length == 0) return false;
	        int m = array.length, n = array[0].length;
	        int i = 0, j = n-1;
	        while(i < m && j >= 0){
	            if(array[i][j] == target) return true;
	            else if(array[i][j] > target) j--;
	            else i++;
	        }
	        return false;
	    }
	}

## 2.替换空格 ##

请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

思路：

字符串操作，遇到' '替换成%20.

	public class Solution {
	    public String replaceSpace(StringBuffer str) {
	    	if(str == null) return null;
	        for(int i = 0; i < str.length(); i++){
	            if(str.charAt(i) == ' '){
	                str.replace(i,i+1,"%20");
	            }
	        }
	        return str.toString();
	    }
	}

## 3.从尾到头打印链表 ##

输入一个链表，按链表从尾到头的顺序返回一个ArrayList。

思路：

DFS

	import java.util.ArrayList;
	public class Solution {
	    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
	        ArrayList<Integer> list = new ArrayList<>();
	        if(listNode == null) return list;
	        helper(list, listNode);
	        return list;
	    }
	    private void helper(ArrayList<Integer> list, ListNode listNode){
	        if(listNode.next != null){
	            helper(list, listNode.next);
	        }
	        list.add(listNode.val);
	    }
	}

## 4.重建二叉树 ##

输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

思路：

前序第一个是根节点，中序该值左边是左子树，右边是右子树。

可以用递归继续构建。

	import java.util.Arrays;
	public class Solution {
	    public TreeNode reConstructBinaryTree(int [] pre,int [] in) {
	        if(pre.length == 0 || in.length == 0) return null;
	        TreeNode root = new TreeNode(pre[0]);
	        for(int i = 0; i < in.length; i++){
	            if(in[i] == pre[0]){
	                root.left = reConstructBinaryTree(Arrays.copyOfRange(pre,1,i+1),Arrays.copyOfRange(in,0,i));
	                root.right = reConstructBinaryTree(Arrays.copyOfRange(pre,i+1,pre.length),Arrays.copyOfRange(in,i+1,in.length));
	                break;
	            }
	        }
	        return root;
	    }
	}

## 5.用两个栈实现队列 ##

用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。

	import java.util.Stack;
	
	public class Solution {
	    Stack<Integer> stack1 = new Stack<Integer>();
	    Stack<Integer> stack2 = new Stack<Integer>();
	    
	    public void push(int node) {
	        stack1.push(node);
	    }
	    
	    public int pop() {
	        if(stack2.isEmpty()){
	            while(!stack1.isEmpty()){
	                stack2.push(stack1.pop());
	            }
	        }
	        return stack2.pop();
	    }
	}

## 6.旋转数组最小数字 ##

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

思路：

当比前一个数小时，为最小元素。特殊情况所有都相等，返回第一个即可。

	import java.util.ArrayList;
	public class Solution {
	    public int minNumberInRotateArray(int [] array) {
	        if(array.length == 0) return 0;
	        for(int i = 1; i < array.length; i++){
	            if(array[i] < array[i-1]) return array[i];
	        }
	        return array[0];
	    }
	}

## 7.斐波那契数列 ##

大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0，第1项是1）。 
n<=39 

思路：

dp[i] = dp[i-1] + dp[i-2]

	public class Solution {
	    public int Fibonacci(int n) {
	        if(n == 0) return 0;
	        if(n == 1) return 1;
	        int pre1 = 0;
	        int pre2 = 1;
	        int result = 0;
	        for(int i = 2; i <= n; i++){
	            result = pre1 + pre2;
	            pre1 = pre2;
	            pre2 = result;
	        }
	        return result;
	    }
	}

## 8.跳台阶 ##

一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

思路：

dp[i] = dp[i-1] + dp[i-2]
	
	public class Solution {
	    public int JumpFloor(int target) {
	        if(target == 1) return 1;
	        if(target == 2) return 2;
	        int pre1 = 1, pre2 = 2;
	        int result = 0;
	        for(int i = 3; i <= target; i++){
	            result = pre1 + pre2;
	            pre1 = pre2;
	            pre2 = result;
	        }
	        return result;
	    }
	}

## 9.变态跳台阶 ##

一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

思路：

每次都可以选择跳与不跳，则跳上n级为2^(n-1)。

	public class Solution {
	    public int JumpFloorII(int target) {
	        if(target == 0) return 0;
	        return (int)Math.pow(2,target-1);
	    }
	}

## 10.矩阵覆盖 ##

我们可以用2 * 1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2 * 1的小矩形无重叠地覆盖一个2 * n的大矩形，总共有多少种方法？ 
 

思路：

dp[i] = dp[i-1] + dp[i-2]

	public class Solution {
	    public int RectCover(int target) {
	        if(target == 1) return 1;
	        if(target == 2) return 2;
	        int pre1 = 1, pre2 = 2;
	        int result = 0;
	        for(int i = 3; i <= target; i++){
	            result = pre1 + pre2;
	            pre1 = pre2;
	            pre2 = result;
	        }
	        return result;
	    }
	}




