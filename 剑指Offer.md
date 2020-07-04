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

注意：每次递归如果检查到最后不符合要求，应把最后一个值删除，从而不影响后续的添加。

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