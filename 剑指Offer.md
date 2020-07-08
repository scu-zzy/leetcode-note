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

递归。

	import java.util.ArrayList;
	public class Solution {
	    ArrayList<Integer> list = new ArrayList<>();
	    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
	        if(listNode == null) return list;
	        if(listNode.next != null) printListFromTailToHead(listNode.next);
	        list.add(listNode.val);
	        return list;
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

## 11.二进制中1的个数 ##

输入一个整数，输出该数32位二进制表示中1的个数。其中负数用补码表示。

思路：

把一个整数减去1，再和原整数做与运算，会把该整数最右边一个1变成0。

	public class Solution {
	    public int NumberOf1(int n) {
	        int count = 0;
	        while(n!=0){
	            count++;
	            n = n&(n-1);
	        }
	        return count;
	    }
	}

## 12.数值的整数次方 ##

给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。 

保证base和exponent不同时为0 

思路：

exponent可能为正可能为负。

	public class Solution {
	    public double Power(double base, int exponent) {
	        //if(exponent == 0) return 1.0;
	        double result = 1.0;
	        for(int i = 0; i < exponent; i++){
	            result *= base;
	        }
	        for(int i = 0; i > exponent; i--){
	            result /= base;
	        }
	        return result;
	  }
	}

## 13.调整数组顺序使奇数位于偶数前面 ##

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

思路：

需要一个辅助数组来保存数据。

	public class Solution {
	    public void reOrderArray(int [] array) {
	        int n = array.length;
	        int[] aux = new int[n];
	        for(int k = 0; k < n; k++){
	            aux[k] = array[k];
	        }
	        int i = 0;
	        for(int k = 0; k < n; k++){
	            if(aux[k]%2 == 1){
	                array[i++] = aux[k];
	            }
	        }
	        for(int k = 0; k < n; k++){
	            if(aux[k]%2 == 0){
	                array[i++] = aux[k];
	            }
	        }
	    }
	}


## 14.链表中倒数第k个结点 ##

输入一个链表，输出该链表中倒数第k个结点。

思路：

双指针。保持两个指针的间隔。

	public class Solution {
	    public ListNode FindKthToTail(ListNode head,int k) {
	        ListNode left = head;
	        ListNode right = left;
	        for(int i = 0; i<k; i++){
				//防止越界
	            if(right == null) return null;
	            right = right.next;
	        }
	        while(right != null){
	            left = left.next;
	            right = right.next;
	        }
	        return left;
	    }
	}


## 15.反转链表 ##

输入一个链表，反转链表后，输出新链表的表头。

思路：

头插法。三个指针保存上一节点，当前节点，下一节点。

	public class Solution {
	    public ListNode ReverseList(ListNode head) {
	        ListNode pre = null;
	        ListNode cur = head;
	        while(cur != null){
	            ListNode next = cur.next;
	            cur.next = pre;
	            pre = cur;
	            cur = next;
	        }
	        return pre;
	    }
	}

## 16.合并两个排序的链表 ##

输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

	/*
	public class ListNode {
	    int val;
	    ListNode next = null;
	
	    ListNode(int val) {
	        this.val = val;
	    }
	}*/
	public class Solution {
	    public ListNode Merge(ListNode list1,ListNode list2) {
	        ListNode result = new ListNode(-1);
	        ListNode head = result;
	        while(list1 != null && list2 != null){
	            if(list1.val < list2.val){
	                result.next = list1;
	                list1 = list1.next;
	            }else{
	                result.next = list2;
	                list2 = list2.next;
	            }
	            result = result.next;
	        }
	        if(list1 == null) result.next = list2;
	        if(list2 == null) result.next = list1;
	        return head.next;
	    }
	}

## 17.树的子结构 ##

输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

思路：

需要构造一个等于的方法。

	public class Solution {
	    public boolean HasSubtree(TreeNode root1,TreeNode root2) {
	        if(root1 == null || root2 == null) return false;
	        return equalTree(root1,root2) || HasSubtree(root1.left,root2) || HasSubtree(root1.right,root2);
	    }
	    private boolean equalTree(TreeNode root1, TreeNode root2){
	        if(root2 == null) return true;
	        if(root1 == null) return false;
	        if(root1.val == root2.val) return equalTree(root1.left,root2.left) && equalTree(root1.right,root2.right);
	        return false;
	    }
	}

## 18.二叉树的镜像 ##

操作给定的二叉树，将其变换为源二叉树的镜像。

	public class Solution {
	    public void Mirror(TreeNode root) {
	        if(root == null) return;
	        TreeNode left = root.left;
	        root.left = root.right;
	        root.right = left;
	        Mirror(root.left);
	        Mirror(root.right);
	    }
	}

## 19.顺时针打印矩阵 ##

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

思路：

定义四个边界。注意：退出应该在每次移动后判定。

	import java.util.ArrayList;
	public class Solution {
	    public ArrayList<Integer> printMatrix(int [][] matrix) {
	        if(matrix.length == 0 || matrix[0].length == 0) return null;
	        int top = 0;
	        int down = matrix.length-1;
	        int left = 0;
	        int right = matrix[0].length-1;
	        ArrayList<Integer> result = new ArrayList<>();
	        while(true){
	            //向右移动
	            for(int i = left; i <= right; i++){
	                result.add(matrix[top][i]);
	            }
	            top++;
	            if(top>down) break;
	            //向下移动
	            for(int i = top; i <= down; i++){
	                result.add(matrix[i][right]);
	            }
	            right--;
	            if(left>right) break;
	            //向左移动
	            for(int i = right; i>= left; i--){
	                result.add(matrix[down][i]);
	            }
	            down--;
	            if(top>down) break;
	            //向上移动
	            for(int i = down; i>= top; i--){
	                result.add(matrix[i][left]);
	            }
	            left++;
	            if(left>right) break;
	        }
	        return result;
	    }
	}

## 20.包含min函数的栈 ##

定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。 

注意：保证测试中不会当栈为空的时候，对栈调用pop()或者min()或者top()方法。

思路：用另一个栈专门储存此时的最小值。

	import java.util.Stack;
	
	public class Solution {
	
	    Stack<Integer> stack = new Stack<>();
	    Stack<Integer> min = new Stack<>();
	    
	    public void push(int node) {
	        stack.push(node);
	        if(min.isEmpty()){
	            min.push(node);
	        }else{
	            //这里不能用pop
	            int minValue = min.peek();
	            minValue = node < minValue ? node : minValue;
	            min.push(minValue);
	        }
	    }
	    
	    public void pop() {
	        stack.pop();
	        min.pop();
	    }
	    
	    public int top() {
	        return stack.peek();
	    }
	    
	    public int min() {
	        return min.peek();
	    }
	}

## 21.栈的压入、弹出序列 ##

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

	public class Solution {
	    public boolean IsPopOrder(int [] pushA,int [] popA) {
	        Stack<Integer> stack = new Stack<>();
	        int l = pushA.length;
	        int j = 0;
	        for(int i = 0; i < l; i++){
	            stack.push(pushA[i]);
	            while(!stack.isEmpty() && stack.peek()==popA[j]){
	                stack.pop();
	                j++;
	            }
	        }
	        return stack.isEmpty();
	    }
	}

## 22.从上到下打印二叉树 ##

从上往下打印出二叉树的每个节点，同层节点从左至右打印。

思路：

BFS。

使用队列进行遍历。

	public class Solution {
	    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
	        ArrayList<Integer> result = new ArrayList<>();
	        if(root == null) return result;
	        Queue<TreeNode> q = new LinkedList<>();
	        q.add(root);
	        while(!q.isEmpty()){
	            TreeNode node = q.poll();
	            result.add(node.val);
	            if(node.left != null) q.add(node.left);
	            if(node.right != null) q.add(node.right);
	        }
	        return result;
	    }
	}

## 23.二叉搜索树的后序遍历序列 ##

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

思路：

最后一个节点为根节点。

小于该节点值的为左子树，大于该节点值的为右子树。

	public class Solution {
	    public boolean VerifySquenceOfBST(int [] sequence) {
	        if(sequence.length == 0) return false;
	        return helper(sequence, 0, sequence.length-1);
	    }
	    private boolean helper(int[] sequence, int start, int root){
	        if(start >= root) return true;
	        int i = start;
	        for(; i < root; i++){
	            if(sequence[i] > sequence[root]) break;
	        }
	        for(int j = i; j < root; j++){
	            if(sequence[j] < sequence[root]) return false;
	        }
	        return helper(sequence, start, i-1) && helper(sequence, i, root-1);
	    }
	}

## 24.二叉树中和为某一值的路径 ##

输入一颗二叉树的根节点和一个整数，按字典序打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。

思路：

回溯。

	public class Solution {
	    ArrayList<ArrayList<Integer>> result = new ArrayList<>();
	    ArrayList<Integer> list = new ArrayList<>();
	    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {
	        if(root == null) return result;
	        list.add(root.val);
	        if(root.left == null && root.right == null && target == root.val){
	            result.add(new ArrayList<Integer>(list));
	        }
	        FindPath(root.left, target - root.val);
	        FindPath(root.right, target - root.val);
	        list.remove(list.size()-1);
	        return result;
	    }
	}

## 25.复杂链表的复制 ##

输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针random指向一个随机节点），请对此链表进行深拷贝，并返回拷贝后的头结点。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

思路：

将链表存入map中，以便之后寻找映射关系。

	import java.util.HashMap;
	public class Solution {
	    public RandomListNode Clone(RandomListNode pHead)
	    {
	        if(pHead == null) return null;
	        RandomListNode head1 = pHead; //记录pHead的头节点
	        HashMap<RandomListNode, RandomListNode> map = new HashMap<>();
	        while(pHead != null){
	            map.put(pHead, new RandomListNode(pHead.label));
	            pHead = pHead.next;
	        }
	        RandomListNode result = new RandomListNode(head1.label);
	        RandomListNode head2 = result;//记录result的头节点
	        while(head1 != null){
	            result.next = map.get(head1.next);
	            result.random = map.get(head1.random);
	            result = result.next;
	            head1 = head1.next;
	        }
	        return head2;
	    }
	}

## 26.二叉搜索树与双向链表 ##

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

思路：

线索化二叉树。类似于中序遍历构建。

由于正向构建时，指针会移动到尾部，因此反向构建。


	public class Solution {
	    TreeNode pre = null;
	    public TreeNode Convert(TreeNode pRootOfTree) {
	        if(pRootOfTree == null) return null;
	        Convert(pRootOfTree.right);
	        if(pre!=null){
	            pRootOfTree.right = pre;
	            pre.left = pRootOfTree;
	        }
	        pre = pRootOfTree;
	        Convert(pRootOfTree.left);
	        return pre;
	    }
	}

## 27.字符串的排列 ##

输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。

> 输入一个字符串,长度不超过9(可能有字符重复),字符只包括大小写字母。

思路：

回溯。

固定A不动，然后交换B与C，从而得到"ABC" 和 "ACB"
同理，对于"BAC"、"BCA" 、"CAB"和"CBA"是同样道理。
当两个字符相同时，不应该交换。

- 递归函数的功能：dfs(int pos, string s), 表示固定字符串s的pos下标的字符s[pos] 
- 递归终止条件：当pos+1 == s.length()的时候，终止，表示对最后一个字符进行固定，也就说明，完成了一次全排列 
- 下一次递归：dfs(pos+1, s), 很显然，下一次递归就是对字符串的下一个下标进行固定

回溯：每次递归完成后，须重新交换回来。

	import java.util.ArrayList;
	import java.util.Collections;
	public class Solution 
	{
	    public ArrayList<String> Permutation(String str)
	    {
	        ArrayList<String> res=new ArrayList<String>();
	        if(str.length()==0||str==null)return res;
	        int n= str.length();
	        helper(res,0,str.toCharArray());
	        Collections.sort(res);
	        return res;
	         
	    }
	    public void helper( ArrayList<String> res,int index,char []s)
	    {
	        if(index==s.length-1)res.add(new String(s));
	        for(int i=index;i<s.length;i++)
	        {
	            if(i==index||s[index]!=s[i])
	            {
	                swap(s,index,i);
	                helper(res,index+1,s);
	                swap(s,index,i);
	            }
	        }
	         
	    }
	     
	    public void swap(char[]t,int i,int j)
	     {
	        char c=t[i];
	        t[i]=t[j];
	        t[j]=c;
	    }
	}

## 28.数组中出现次数超过一半的数字 ##

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

思路：

使用HashMap。

	import java.util.HashMap;
	public class Solution {
	    public int MoreThanHalfNum_Solution(int [] array) {
	        if(array.length == 0) return 0;
	        HashMap<Integer, Integer> map = new HashMap<>();
	        for(int val : array){
	            map.put(val,map.getOrDefault(val,0)+1);
	        }
	        for(int key : map.keySet()){
	            if(map.get(key)*2 > array.length) return key;
	        }
	        return 0;
	    }
	}

## 29.最小K个数 ##

输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。

排序：

	import java.util.*;
	public class Solution {
	    public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
	        ArrayList<Integer> result = new ArrayList<>();
	        if(input.length == 0 || k > input.length) return result;
	        Arrays.sort(input);
	        for(int i = 0; i < k; i++){
	            result.add(input[i]);
	        }
	        return result;
	    }
	}

使用堆：

	import java.util.*;
	public class Solution {
	    public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
	        ArrayList<Integer> result = new ArrayList<>();
	        if(input.length == 0 || k > input.length) return result;
	        PriorityQueue<Integer> heap = new PriorityQueue<>();
	        for(int value : input){
	            heap.add(value);
	        }
	        for(int i = 0; i < k; i++){
		        result.add(heap.poll());
		    }
		    return result;
	    }
	}

## 30.连续子数组的最大和 ##
HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)

思路：

需要两个变量，一个储存当前连续子向量的和，一个储存最大值。

	public class Solution {
	    public int FindGreatestSumOfSubArray(int[] array) {
	        int cur = array[0], max = array[0];
	        for(int i = 1; i < array.length; i++){
	            cur = cur + array[i];
	            cur = Math.max(cur, array[i]);
	            max = Math.max(max, cur);
	        }
	        return max;
	    }
	}

## 31.整数中1出现的次数 ##

求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

思路：

将一个数字中1出现的次数拆成个、十、百位中1出现次数的和以321为例：  

- cnt = 32 + 1，把321拆成高位32和1，固定个位是1，高位的取值可以是0~31共32个数，由于低位为1大于0，所以高位还可以取32（即数字321），则个位上1出现的次数是32+1=33 
- cnt = 30 + 10，把321拆成高位3和21，固定十位是1，高位可以取 0 ~ 2 共30个数，由于低位是21-10+1大于0，所以高位还可以取3（即数字310~319），则十位上1出现的次数是30 + 10 = 40 
- cnt = 0 + 100，把321拆成高位0和321，固定百位是1，高位可以取 0 个数，由于低位是321-100+1大于0，所以可以取数字100~199），则百位上1出现的次数是0 + 100 = 100  

所以321中1出现的次数是173

	public class Solution {
	    public int NumberOf1Between1AndN_Solution(int n) {
	        int cnt = 0, i = 1;
	        while(i<=n){
	            cnt += n / (i * 10) * i + Math.min(Math.max(n % (i * 10) - i + 1, 0), i);
	            i *= 10;
	        }
	        return cnt;
	    }
	}

## 32.把数组排成最小的数 ##

输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

思路：

贪心，如果ab>ba，则应该将b排在前面。因此可以自定义排序规则。

	import java.util.*;
	
	public class Solution {
	    public String PrintMinNumber(int [] numbers) {
	        if(numbers.length == 0) return "";
	        Integer[] numberObjects = new Integer[numbers.length];
	        for(int i = 0; i < numbers.length; i++){
	            numberObjects[i] = numbers[i];
	        }
	        Arrays.sort(numberObjects,new Comparator<Integer>(){
	            public int compare(Integer a, Integer b){
	                return Integer.valueOf(""+a+b)-Integer.valueOf(""+b+a);
	            }
	        });
	        StringBuilder result = new StringBuilder();
	        for(int numberObject : numberObjects){
	            result.append(numberObject);
	        }
	        return result.toString();
	    }
	}

## 33.丑数 ##

把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

思路：

每个丑数乘2，3，5都是丑数，每次记录最小的那个丑数,则该丑数序列是有序的。

如果x=y=z那么最小丑数一定是乘以2的，但关键是有可能存在x>y>z的情况，所以我们要维持三个指针来记录当前乘以2、乘以3、乘以5的最小值，然后当其被选为新的最小值后，要把相应的指针+1；

	public class Solution {
	    public int GetUglyNumber_Solution(int index) {
	        if(index<=0) return 0;
	        int[] result = new int[index];
	        result[0] = 1;
	        int p2 = 0, p3 = 0, p5 = 0;
	        for(int i = 1; i < index; i++){
	            result[i] = Math.min(result[p2]*2, Math.min(result[p3]*3, result[p5]*5));
	            if(result[i] == result[p2]*2) p2++;
	            if(result[i] == result[p3]*3) p3++;
	            if(result[i] == result[p5]*5) p5++;
	        }
	        return result[index-1];
	    }
	}

## 34.第一次只出现一次的字符 ##

在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.（从0开始计数）

思路：

HashMap

	import java.util.*;
	public class Solution {
	    public int FirstNotRepeatingChar(String str) {
	        if(str.length()==0) return -1;
	        Map<Character, Integer> map = new HashMap<>();
	        for(char c : str.toCharArray()){
	            map.put(c, map.getOrDefault(c,0)+1);
	        }
	        for(int i = 0; i < str.length(); i++){
	            if(map.get(str.charAt(i)) == 1){
	                return i;
	            }
	        }
	        return -1;
	    }
	}

## 35.数组的逆序对 ##

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007

思路：

归并排序的思想。

对于本题，在两个子序列left、right合并过程中，当left中当前元素A大于right中当前元素B时，因为left序列已经有序，所以A后面所有元素都可以与B组成逆序对。

	public class Solution {
	    int[] aux;
	    int count = 0;
	    public int InversePairs(int [] array) {
	        aux = new int[array.length];
	        mergeSort(array, 0, array.length-1);
	        return count;
	    }
	    private void merge(int[] array, int l, int m, int h){
	        int i = l, j = m+1;
	        for(int k = l; k <= h; k++){
	            aux[k] = array[k];
	        }
	        for(int k = l; k <= h; k++){
	            if(i > m){
	                array[k] = aux[j++];
	            }else if(j > h){
	                array[k] = aux[i++];
	            }else if(aux[i] <= aux[j]){
	                array[k] = aux[i++];
	            }else{
	                array[k] = aux[j++];
	                count = (count + m + 1 - i)%1000000007;
	            }
	        }
	    }
	    private void mergeSort(int[] array, int l, int h){
	        if(l>=h) return;
	        int m = l + (h-l)/2;
	        mergeSort(array,l,m);
	        mergeSort(array,m+1,h);
	        merge(array,l,m,h);
	    }
	}

## 36.两个链表的第一个公共节点 ##

输入两个链表，找出它们的第一个公共结点。（注意因为传入数据是链表，所以错误测试数据的提示是用其他方式显示的，保证传入数据是正确的）

思路：

双指针，a+c+b=b+c+a，所以两个指针将两个链表遍历完后，遍历另一个链表能够相遇，相遇处就是公共节点，没有公共节点则恰好返回null。

	public class Solution {
	    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
	        if(pHead1 == null || pHead2 == null) return null;
	        ListNode p1 = pHead1, p2 = pHead2;
	        while(p1 != p2){
	            p1 = (p1 == null) ? pHead2 : p1.next;
	            p2 = (p2 == null) ? pHead1 : p2.next;
	        }
	        return p1;
	    }
	}

## 37.统计一个数字在排序数组中出现的次数。 ##

统计一个数字在排序数组中出现的次数。

思路：

二分查找。找到该数的最左位置。

	public class Solution {
	    public int GetNumberOfK(int [] array , int k) {
	        if(array.length == 0) return 0;
	        int count = 0;
	        int l = 0, h = array.length - 1;
	        while(l<h){
	            int m = l + (h-l)/2;
	            if(array[m] >= k){
	                h = m;
	            }else{
	                l = m + 1;
	            }
	        }
	        while(l < array.length && array[l] == k){
	            count++;
	            l++;
	        }
	        return count;
	    }
	}

## 38.二叉树的深度 ##

输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

思路：

递归。

	public class Solution {
	    public int TreeDepth(TreeNode root) {
	        if(root == null) return 0;
	        return 1 + Math.max(TreeDepth(root.left), TreeDepth(root.right)); 
	    }
	}

## 39.平衡二叉树 ##

输入一棵二叉树，判断该二叉树是否是平衡二叉树。 

在这里，我们只需要考虑其平衡性，不需要考虑其是不是排序二叉树 

思路：

递归。

	public class Solution {
	    boolean flag = true;
	    public boolean IsBalanced_Solution(TreeNode root) {
	        if(root == null) return true;
	        deep(root);
	        return flag;
	    }
	    public int deep(TreeNode root){
	        if(root == null) return 0;
	        if(Math.abs(deep(root.left)-deep(root.right))>1) flag = false;
	        return 1 + Math.max(deep(root.left), deep(root.right)); 
	    }
	}

## 40.数组中只出现一次的数字 ##

一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。

思路：

位运算。

	public class Solution {
	    public void FindNumsAppearOnce(int [] array,int num1[] , int num2[]) {
	        int diff = 0;
	        for(int num : array){
	            diff ^= num;
	        }
	        //取最右边不为0的位来取分两数
	        diff &= -diff;
	        for(int num : array){
	            if((num & diff) == 0) num1[0] ^= num;
	            else num2[0] ^= num;
	        }
	    }
	}

## 41.和为S的连续正数序列 ##

小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!

思路：

遍历。注意边界问题。

	import java.util.ArrayList;
	public class Solution {
	    public ArrayList<ArrayList<Integer> > FindContinuousSequence(int sum) {
	        ArrayList<ArrayList<Integer> > result = new ArrayList<>();
	        for(int i = 1; i <= sum/2; i++){
	            int s = 0;
	            ArrayList<Integer> list = new ArrayList<>();
	            for(int j = i; j <= sum/2 + 1 && s < sum; j++){
	                list.add(j);
	                s += j;
	                if(s == sum) result.add(list);
	            }
	        }
	        return result;
	    }
	}

## 42.和为S的两个数字 ##

输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

思路：

双指针。

	import java.util.ArrayList;
	public class Solution {
	    public ArrayList<Integer> FindNumbersWithSum(int [] array,int sum) {
	        ArrayList<Integer> list = new ArrayList<>();
	        int l = 0, r = array.length - 1;
	        while(l < r){
	            int s = array[l] + array[r];
	            if(s == sum){
	                list.add(array[l]);
	                list.add(array[r]);
	                break;
	            }else if(s > sum) r--;
	            else l++;
	        }
	        return list;
	    }
	}

## 43.左旋转字符串 ##

汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！

思路：

substring

	public class Solution {
	    public String LeftRotateString(String str,int n) {
	        if(str.length()==0) return str;
	        n = n % str.length();
	        String left = str.substring(0,n);
	        String right = str.substring(n,str.length());
	        return right + left;
	    }
	}

## 44.翻转单词顺序序列 ##

牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？

思路：

split。

	public class Solution {
	    public String ReverseSentence(String str) {
	        String[] array = str.split(" ");
	        if(array.length == 0) return str;
	        StringBuilder sb = new StringBuilder();
	        for(int i = array.length-1; i > 0; i--){
	            sb.append(array[i]).append(" ");
	        }
	        sb.append(array[0]);
	        return sb.toString();
	    }
	}

## 45.扑克牌顺子 ##

LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。

思路：

最大的数减最小但不是0的数应该小于等于4且不能有重复.

则使用HashSet来解决重复问题。

	import java.util.*;
	public class Solution {
	    public boolean isContinuous(int [] numbers) {
	        int max = 0;
	        int min = 14;
	        int count = 0;
	        HashSet<Integer> set = new HashSet<>();
	        for(int number : numbers){
	            if(number == 0){ 
	                count++;
	                continue;
	            }
	            set.add(number);
	            min = Math.min(min, number);
	            max = Math.max(max, number);
	        } 
	        if(count+set.size()<5) return false;
	        if(max - min > 4) return false;
	        return true;
	    }
	}

## 46.圆圈中最后剩下的数 ##

每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1) 

如果没有小朋友，请返回-1 

思路：

使用List模拟，使用一个指针。

	import java.util.*;
	public class Solution {
	    public int LastRemaining_Solution(int n, int m) {
	        if(n == 0 || m == 0) return -1;
	        LinkedList<Integer> list = new LinkedList<>();
	        for(int i = 0; i < n; i++) list.add(i);
	        int cur = -1;
	        while(list.size()>1){
	            for(int i = 0; i < m; i++){
	                cur++;
	                if(cur == list.size()) cur = 0;
	            }
	            list.remove(cur);
	            //删除节点时，cur会指向下一个节点，因此需要减1。
	            cur--;
	        }
	        return list.get(0);
	    }
	}

## 47.求1+2+3+...+n ##

求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

	public class Solution {
	    public int Sum_Solution(int n) {
	        //当n=0时，&&后面的就不会执行了，直接返回0
	        boolean flag = n>0 && (n += Sum_Solution(n-1)) > 0; 
	        return n;
	    }
	}

## 48.不用加减乘除做加法 ##

思路：

位运算

不考虑进位的加法： sum = num1 ^ num2;

进位：carry = (num1 & num2) << 1;

	public class Solution {
	    public int Add(int num1,int num2) {
	        int sum = 0;
	        int carry = 1;
	        while(carry!=0){
	            sum = num1 ^ num2;
	            carry = (num1 & num2) << 1;
	            num1 = sum;
	            num2 = carry;
	        }
	        return sum;
	    }
	}

## 49.把字符串转换为整数 ##

将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0

思路：

注意符号与边界。

	public class Solution {
	    public int StrToInt(String str) {
	        if(str.length() == 0) return 0;
	        int flag = 1;//表示符号，正数为1，负数为-1
	        int i = 0;//指针，用于记录第一位是否为符号
	        int result = 0;
	        if(str.charAt(0) == '-') flag = -1;
	        if(str.charAt(0) == '-' || str.charAt(0) == '+') i++;
	        for(int j = i; j < str.length(); j++){
	            char num = str.charAt(j);
	            if(isNum(num)){
	                int cur = num - '0';
	                //正数的边界如下
	                if(flag > 0 && (result > Integer.MAX_VALUE/10 || (result == Integer.MAX_VALUE/10 && cur > 7))){
	                    return 0;
	                }
	                //负数的边界如下
	                if(flag < 0 && (result > Integer.MAX_VALUE/10 || (result == Integer.MAX_VALUE/10 && cur > 8))){
	                    return 0;
	                }
	                result = result*10 + cur;
	            }else{
	                return 0;
	            }
	        }
	        return flag*result;
	    }
	    private boolean isNum(char c){
	        return c >= '0' && c <= '9';
	    }
	}

## 50.数组中重复的数字 ##

在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

思路：

使用HashMap，遇到重复数字就返回。

	import java.util.*;
	public class Solution {
	    // Parameters:
	    //    numbers:     an array of integers
	    //    length:      the length of array numbers
	    //    duplication: (Output) the duplicated number in the array number,length of duplication array is 1,so using duplication[0] = ? in implementation;
	    //                  Here duplication like pointor in C/C++, duplication[0] equal *duplication in C/C++
	    //    这里要特别注意~返回任意重复的一个，赋值duplication[0]
	    // Return value:       true if the input is valid, and there are some duplications in the array number
	    //                     otherwise false
	    public boolean duplicate(int numbers[],int length,int [] duplication) {
	        if(length <= 0) return false;
	        HashMap<Integer, Integer> map = new HashMap<>();
	        for(int i = 0; i < length; i++){
	            map.put(numbers[i], map.getOrDefault(numbers[i], 0)+1);
	            if(map.get(numbers[i]) == 2){
	                duplication[0] = numbers[i];
	                return true;
	            }
	        }
	        return false;
	    }
	}

## 51.构建乘积数组 ##

给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。（注意：规定B[0] = A[1] * A[2] * ... * A[n-1]，B[n-1] = A[0] * A[1] * ... * A[n-2];）


思路：

假设：

	left[i] = A[0]*...*A[i-1]
	right[i] = A[i+1]*...*A[n-1]
所以：

	B[i] = left[i] * right[i]

可知：

	left[i+1] = left[i] * A[i]
	right[i] = right[i+1] * A[i+1]

B[0]没有左，B[n-1]没有右。

	import java.util.Arrays;
	public class Solution {
	    public int[] multiply(int[] A) {
	        int n = A.length;
	        int[] B = new int[n];
	        if(n == 0) return B;
	        Arrays.fill(B,1);
	        for(int i = 1; i < n; i++){
	            B[i] = B[i-1]*A[i-1];
	        }
	        int temp = 1;
	        for(int i = n-2; i >= 0; i--){
	            temp *= A[i+1];
	            B[i] *= temp;
	        }
	        return B;
	    }
	}

## 52.正则表达式匹配 ##

请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配


思路：

递归。使用两个指针记录两个字符串的匹配位置。

若pattern下一个字符有' * '，如果该字符匹配或者此位置为' . '，str指针向右移动1位（表示该字符可能出现多于一次），或者pattern指针向右移动两位（表示该字符出现0次）； 如果字符不匹配，那么该字符肯定出现0次，pattern指针向右移动两位。

若pattern下一个字符没有' * '， 也是分为上面两种情况。

	public class Solution {
	    public boolean match(char[] str, char[] pattern)
	    {
	        return match(str, 0, pattern, 0);
	    }
	    private boolean match(char[] str, int strIndex, char[] pattern, int patternIndex){
	        //两个字符串同时检查完
	        if(strIndex == str.length && patternIndex == pattern.length){
	            return true;
	        //pattren先检查完
	        }else if(patternIndex == pattern.length){
	            return false;
	        }
	        //判断pattern下一个字符有'*'，从而分为两种情况
	        boolean flag = (patternIndex < pattern.length - 1) && (pattern[patternIndex+1] == '*');
	        if(flag){
	            //如果该字符匹配或者此位置为'.'
	            //注意是在这里判断strIndex是否越界，因为str为空时可以与".*"匹配
	            if(strIndex < str.length && (pattern[patternIndex] == '.' || str[strIndex] == pattern[patternIndex])){
	                //分两种情况，前者表示'*'前的字符可能出现多于一次，后者表示该字符出现0次。
	                return match(str, strIndex+1, pattern, patternIndex) || match(str, strIndex, pattern, patternIndex+2);
	            }else{//该字符不匹配
	                //这时只有一种可能，那就是'*'前的字符出现0次。
	                return match(str, strIndex, pattern, patternIndex+2);
	            }
	        }else{//同样分上面两种情况
	            if(strIndex < str.length && (pattern[patternIndex] == '.' || str[strIndex] == pattern[patternIndex])){
	                //此时，对应字符相匹配，那么检查下一个字符是否匹配
	                return match(str, strIndex+1, pattern, patternIndex+1);
	            }else{
	                //该字符不匹配则整个字符串不匹配
	                return false;
	            }
	        }
	    }
	}

## 53.表示数值的字符串 ##

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。

思路：

使用正则表达式来匹配。

	import java.util.regex.Pattern;
	public class Solution {
	    public boolean isNumeric(char[] str) {
	        String pattern="^[-+]?\\d*(\\.\\d*)?([eE][-+]?\\d+)?$";
	        String s = String.valueOf(str);
	        return Pattern.matches(pattern,s);
	    }
	}


## 54.字符流中第一个不重复的字符 ##

请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。

思路：

哈希和队列实现。

哈希记录次数，队列记录顺序。

	import java.util.*;
	public class Solution {
	    Queue<Character> queue = new LinkedList<>();
	    Map<Character, Integer> map = new HashMap<>();
	    //Insert one char from stringstream
	    public void Insert(char ch)
	    {
	        queue.add(ch);
	        map.put(ch, map.getOrDefault(ch, 0)+1);
	    }
	  //return the first appearence once char in current stringstream
	    public char FirstAppearingOnce()
	    {
	        while(queue.peek() != null){
	            Character c = queue.peek();
	            if(map.get(c) == 1) return c;
	            //只有重复的才会出列，不重复会一直保存
	            else queue.poll();
	        }
	        return '#';
	    }
	}


## 55.链表中环的入口结点 ##

给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。

思路：

快慢指针。

1. 初始化：快指针fast指向头结点， 慢指针slow指向头结点 
1. 让fast一次走两步， slow一次走一步，第一次相遇在C处，停止 
1. 然后让fast指向头结点，slow原地不动，让后fast，slow每次走一步，当再次相遇，就是入口结点。

		public class Solution {
		
		    public ListNode EntryNodeOfLoop(ListNode pHead)
		    {
		        if(pHead == null || pHead.next == null || pHead.next.next == null) return null;
		        ListNode p1 = pHead.next, p2 = pHead.next.next;
		        while(p1 != p2){
		            if(p1.next == null || p2.next.next == null) return null;
		            p1 = p1.next;
		            p2 = p2.next.next;
		        }
		        p2 = pHead;
		        while(p1 != p2){
		            p1 = p1.next;
		            p2 = p2.next;
		        }
		        return p2;
		    }
		}
   
## 56.删除链表中重复的结点 ##

在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

思路：

遇到下一个是重复节点则一直向前走直到不是重复节点。

	public class Solution {
	    public ListNode deleteDuplication(ListNode pHead)
	    {
	        if(pHead == null || pHead.next==null) return pHead;
	        ListNode cur = pHead, next = cur.next;
	        ListNode pre = new ListNode(-1);
	        pre.next = pHead;
	        ListNode result = pre;
	        while(cur != null){
	            next = cur.next;
	            if(next == null) break;
	            //不相等先判断
	            if(cur.val != next.val){
	                pre = cur;
	                cur = next;
	                continue;
	            }
	            //这里空判断应该先判断
	            while(next != null && cur.val == next.val){
	                next = next.next;
	            }
	            pre.next = next;
	            cur = next;
	        }
	        return result.next;
	    }
	}

## 57.二叉树的下一个结点 ##

给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

思路：

中序遍历，寻找。

应先找到根节点。

	import java.util.*;
	public class Solution {
	    List<TreeLinkNode> list = new ArrayList<>();
	    public TreeLinkNode GetNext(TreeLinkNode pNode)
	    {
	        if(pNode == null) return null;
	        TreeLinkNode root = pNode;
	        while(root.next != null){
	            root = root.next;
	        }
	        inorder(root);
	        for(int i = 0; i < list.size()-1; i ++){
	            if(list.get(i) == pNode) return list.get(i+1);
	        }
	        return null;
	    }
	    private void inorder(TreeLinkNode root){
	        if(root == null) return;
	        inorder(root.left);
	        list.add(root);
	        inorder(root.right);
	    }
	}

 
## 58.对称二叉树 ##   

请实现一个函数，用来判断一棵二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

思路：

左子树的左子树和右子树的右子树相等且左子树的右子树和右子树的左子树相等。

	public class Solution {
	    boolean isSymmetrical(TreeNode pRoot)
	    {
	        if(pRoot == null) return true;
	        return isSymmetrical(pRoot.left, pRoot.right);
	    }
	    boolean isSymmetrical(TreeNode left, TreeNode right){
	        if(left == null && right == null) return true;
	        if(left == null || right == null) return false;
	        if(left.val != right.val) return false;
	        return isSymmetrical(left.left, right.right) && isSymmetrical(left.right, right.left);
	    }
	}

## 59.按之字形打印二叉树 ##

请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

思路：

BFS，使用队列，用一个标志位来决定是否需要翻转。

	import java.util.*;
	public class Solution {
	    public ArrayList<ArrayList<Integer> > Print(TreeNode pRoot) {
	        ArrayList<ArrayList<Integer> > result = new ArrayList<>();
	        if(pRoot == null) return result;
	        Queue<TreeNode> queue = new LinkedList<>();
	        queue.add(pRoot);
	        boolean reverse = false;
	        while(!queue.isEmpty()){
	            ArrayList<Integer> list = new ArrayList<>();
	            int size = queue.size();
	            for(int i = 0; i < size; i++){
	                TreeNode node = queue.poll();
	                list.add(node.val);
	                if(node.left != null) queue.add(node.left);
	                if(node.right != null) queue.add(node.right);
	            }
	            if(reverse){
	                Collections.reverse(list);
	            }
	            result.add(list);
	            reverse = !reverse;
	        }
	        return result;
	    }
	}

## 60.把二叉树打印成多行 ##

从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。

思路：

与上题相似，此时不需要翻转。

	import java.util.*;
	public class Solution {
	    public ArrayList<ArrayList<Integer> > Print(TreeNode pRoot) {
	        ArrayList<ArrayList<Integer> > result = new ArrayList<>();
	        if(pRoot == null) return result;
	        Queue<TreeNode> queue = new LinkedList<>();
	        queue.add(pRoot);
	        while(!queue.isEmpty()){
	            ArrayList<Integer> list = new ArrayList<>();
	            int size = queue.size();
	            for(int i = 0; i < size; i++){
	                TreeNode node = queue.poll();
	                list.add(node.val);
	                if(node.left != null) queue.add(node.left);
	                if(node.right != null) queue.add(node.right);
	            }
	            result.add(list);
	        }
	        return result;
	    }
	}

## 61.序列化二叉树 ##

请实现两个函数，分别用来序列化和反序列化二叉树 

二叉树的序列化是指：把一棵二叉树按照某种遍历方式的结果以某种格式保存为字符串，从而使得内存中建立起来的二叉树可以持久保存。序列化可以基于先序、中序、后序、层序的二叉树遍历方式来进行修改，序列化的结果是一个字符串，序列化时通过 某种符号表示空节点（#），以 ！ 表示一个结点值的结束（value!）。

二叉树的反序列化是指：根据某种遍历顺序得到的序列化字符串结果str，重构二叉树。

例如，我们可以把一个只有根节点为1的二叉树序列化为"1,"，然后通过自己的函数来解析回这个二叉树

	public class Solution {
	    int index = -1;
	    String Serialize(TreeNode root) {
	        StringBuilder sb = new StringBuilder();
	        if(root == null) return "#,";
	        sb.append(root.val).append(",");
	        sb.append(Serialize(root.left));
	        sb.append(Serialize(root.right));
	        return sb.toString();
	  }
	    TreeNode Deserialize(String str) {
	        index++;
	        String[] strs = str.split(",");
	        if(strs.length <= index) return null;
	        if(strs[index].equals("#")) return null;
	        TreeNode root = new TreeNode(Integer.valueOf(strs[index]));
	        root.left = Deserialize(str);
	        root.right = Deserialize(str);
	        return root;
	  }
	}

## 62.二叉搜索树的第K个节点 ##

给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。

思路：

中序遍历第k个。

	public class Solution {
	    TreeNode val;
	    int index = 0;
	    TreeNode KthNode(TreeNode pRoot, int k)
	    {
	        if(pRoot == null) return null;
	        KthNode(pRoot.left, k);
	        index++;
	        if(index == k) val = pRoot;
	        KthNode(pRoot.right, k);
	        return val;
	    }
	}

## 63.数据流中的中位数 ##

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。

	import java.util.*;
	public class Solution {
	    
	    ArrayList<Integer> list = new ArrayList<>();
	
	    public void Insert(Integer num) {
	        if(num == null) return;
	        list.add(num);
	    }
	
	    public Double GetMedian() {
	        int n = list.size();
	        if(n == 0) return 0.0;
	        Collections.sort(list);
	        if(n % 2 == 1){
	            return list.get(n/2)/1.0;
	        }else{
	            return (list.get(n/2-1) + list.get(n/2))/2.0;
	        }
	    }
	}

## 64.滑动窗口的最大值 ##

给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。

思路：

使用大顶堆。

	import java.util.*;
	public class Solution {
	    public ArrayList<Integer> maxInWindows(int [] num, int size)
	    {
	        //大顶堆
	        PriorityQueue<Integer> maxQueue = new PriorityQueue<>(new Comparator<Integer>(){
	            public int compare(Integer a, Integer b) {return b - a;}
	        });
	        
	        ArrayList<Integer> result = new ArrayList<>();
	        
	        if(size > num.length || size <= 0 || num.length == 0) return result;
	        
	        //初始化大顶堆
	        int count = 0;
	        while(count < size) maxQueue.add(num[count++]);
	        
	        while(count < num.length){
	            result.add(maxQueue.peek());
	            maxQueue.remove(num[count-size]);
	            maxQueue.add(num[count++]);
	        }
	        result.add(maxQueue.peek());//最后一次入堆没保存结果，额外做一次
	        return result;
	    }
	}

## 65.矩阵中的路径 ##

请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。 

思路：

回溯。

	public class Solution {
	    public boolean hasPath(char[] matrix, int rows, int cols, char[] str)
	    {
	        //标志位，初始化为false
	        boolean[] flag = new boolean[matrix.length];
	        for(int i=0;i<rows;i++){
	            for(int j=0;j<cols;j++){
	                 //循环遍历二维数组，找到起点等于str第一个元素的值，再递归判断四周是否有符合条件的----回溯法
	                 if(judge(matrix,i,j,rows,cols,flag,str,0)){
	                     return true;
	                 }
	            }
	        }
	        return false;
	    }
	     
	    //judge(初始矩阵，索引行坐标i，索引纵坐标j，矩阵行数，矩阵列数，待判断的字符串，字符串索引初始为0即先判断字符串的第一位)
	    private boolean judge(char[] matrix,int i,int j,int rows,int cols,boolean[] flag,char[] str,int k){
	        //先根据i和j计算匹配的第一个元素转为一维数组的位置
	        int index = i*cols+j;
	        //递归终止条件
	        if(i<0 || j<0 || i>=rows || j>=cols || matrix[index] != str[k] || flag[index] == true)
	            return false;
	        //若k已经到达str末尾了，说明之前的都已经匹配成功了，直接返回true即可
	        if(k == str.length-1)
	            return true;
	        //要走的第一个位置置为true，表示已经走过了
	        flag[index] = true;
	         
	        //回溯，递归寻找，每次找到了就给k加一，找不到，还原
	        if(judge(matrix,i-1,j,rows,cols,flag,str,k+1) ||
	           judge(matrix,i+1,j,rows,cols,flag,str,k+1) ||
	           judge(matrix,i,j-1,rows,cols,flag,str,k+1) ||
	           judge(matrix,i,j+1,rows,cols,flag,str,k+1)  )
	        {
	            return true;
	        }
	        //走到这，说明这一条路不通，还原，再试其他的路径
	        flag[index] = false;
	        return false;
	    }
	}

## 66.机器人的运动范围 ##

地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？

思路：

搜索。需要标记走过的路线。

	public class Solution {
	
	    // 判断坐标是否符合要求
	    public boolean isValid(int row, int col, int threshold){
	        int sum = 0;
	        while(row > 0){
	            sum += row%10;
	            row = row/10;
	        }
	        while(col > 0){
	            sum += col%10;
	            col = col/10;
	        }
	        if(sum > threshold)return false;
	        else return true;
	    }
	    //统计能够走到的次数
	    public int count = 0;
	
	    public void help(int i, int j, int threshold, int rows, int cols, int[][] flag){
	        if(i < 0 || i >= rows || j < 0 || j >= cols)return;//如果坐标不符合则不计数
	        if(flag[i][j] == 1)return;//如果曾经被访问过则不计数
	        if(!isValid(i, j, threshold)){
	            flag[i][j] = 1;//如果不满足坐标有效性，则不计数并且标记是访问的
	            return;
	        }
	        //无论是广度优先遍历还是深度优先遍历，我们一定要知道的时候遍历一定会有终止条件也就是要能够停止，
	        //不然程序就会陷入死循环，这个一定是我们做此类题目必须要注意的地方
	        flag[i][j] = 1;
	        count++;
	         //向上，此题目不需要
	        //help(i-1, j, threshold, rows, cols, flag);//遍历上下左右节点
	        //向下
	        help(i+1, j, threshold, rows, cols, flag);
	        //向左，此题目不需要
	        //help(i, j-1, threshold, rows, cols, flag);
	        //向右
	        help(i, j+1, threshold, rows, cols, flag);
	    }
	
	
	    public int movingCount(int threshold, int rows, int cols)
	    {
	        int[][] flag = new int[rows][cols];
	        help(0, 0, threshold, rows, cols, flag);
	        return count;
	    }
	}

## 67.剪绳子 ##

给你一根长度为n的绳子，请把绳子剪成整数长的m段（m、n都是整数，n>1并且m>1，m<=n），每段绳子的长度记为k[1],...,k[m]。请问k[1]x...xk[m]可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

	public class Solution {
	    public int cutRope(int target) {
	        int[] dp = new int[target+1];
	        dp[1] = 1;
	        for(int i = 2; i <= target; i++){
	            for(int j = 1; j < i; j++){
	                dp[i] = Math.max(dp[i], Math.max(j*dp[i-j], j*(i-j)));
	            }
	        }
	        return dp[target];
	    }
	}

