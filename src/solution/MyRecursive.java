package solution;

import javafx.util.Pair;
import tree.BinaryNode;

import java.util.*;

public class MyRecursive {
    /**
     * 问题1：以相反的顺序打印字符串
     * 解决办法：递归实现
     * @param str 给定的字符串
     */
    private static void printReverse(char [] str) {
        printReverseHelper(0, str);
    }
    private static void printReverseHelper(int index, char [] str) {
        if (str == null || index >= str.length) {
            return;
        }
        printReverseHelper(index + 1, str);
        System.out.print(str[index]);
    }

    /**
     * 问题2：反转字符串
     * 具体描述：编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。
     *         不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
     *         你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符。
     * 示例 1：
     * 输入：["h","e","l","l","o"]
     * 输出：["o","l","l","e","h"]
     *
     * 示例 2：
     * 输入：["H","a","n","n","a","h"]
     * 输出：["h","a","n","n","a","H"]
     *
     * 解决办法：循环调换
     *
     * @param s 给定字符串的字符数组
     */
    public static void reverseString(char[] s) {
        if (s == null) return ;

        int length = s.length;
        if (length < 2) return ;

        for (int i = 0; i < length / 2; i++) {
            int changeIndex = length - 1 - i;
            char tmp = s[i];
            s[i] = s[changeIndex];
            s[changeIndex] = tmp;
        }
    }
    /**
     * 问题2：反转字符串
     * 解决办法：递归调换
     * @param s 给定字符串的字符数组
     */
    public static void reverseStringByRecursive(char[] s) {
        reverseStringByRecursiveHelper(0, s.length - 1, s);
    }
    private static void reverseStringByRecursiveHelper(int start, int end, char [] s) {
        if (start >= end) {
            return;
        }
        // swap between the first and the last elements.
        char tmp = s[start];
        s[start] = s[end];
        s[end] = tmp;

        reverseStringByRecursiveHelper(start + 1, end - 1, s);
    }

    /*
    按照我们上面列出的步骤，我们可以按下面的流程来实现函数：
    首先，我们交换列表中的前两个节点，也就是 head 和 head.next；
    然后我们以 swap(head.next.next) 的形式调用函数自身，以交换头两个节点之后列表的其余部分。
    最后，我们将步骤（2）中的子列表的返回头与步骤（1）中交换的两个节点相连，以形成新的链表。
     */
    static class ListNode {
        int val;
        ListNode next;
        ListNode(int x) { val = x; }

        @Override
        public String toString() {
            StringBuilder res = new StringBuilder();
            res.append("[");
            ListNode cur = this;
            while (cur != null) {
                res.append(cur.val);
                res.append(",");
                cur = cur.next;
            }
            res.deleteCharAt(res.length() - 1);
            res.append("]");
            return res.toString();
        }
    }
    /**
     * 问题3：两两交换链表中的节点
     * 具体描述：给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
     *         你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
     * 示例:
     * 给定 1->2->3->4, 你应该返回 2->1->4->3.
     *
     * @param head 链表头节点
     * @return 交换后的头节点
     */
    public static ListNode swapPairs(ListNode head) {
        if (head == null) return null;

        return swapTwoNode(head, head.next);
    }
    public static ListNode swapTwoNode(ListNode cur, ListNode next) {
        if (cur == null || next == null) return cur;

        ListNode nn = next.next;
        next.next = cur;
        cur.next = nn;

        if (cur.next != null)
            cur.next = swapTwoNode(cur.next, cur.next.next);

        return next;
    }

    /**
     * 问题4：杨辉三角
     * 具体描述：给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。
     *
     * 示例:
     * 输入: 5
     * 输出:
     * [
     *      [1],
     *     [1,1],
     *    [1,2,1],
     *   [1,3,3,1],
     *  [1,4,6,4,1]
     * ]
     *
     * 解决方案：循环
     *
     * @param numRows 给定行数
     * @return 杨辉三角集合
     */
    public static List<List<Integer>> generatePascalTriang(int numRows) {
        List<List<Integer>> triangle = new ArrayList<List<Integer>>();

        // First base case; if user requests zero rows, they get zero rows.
        if (numRows == 0) {
            return triangle;
        }

        // Second base case; first row is always [1].
        triangle.add(new ArrayList<>());
        triangle.get(0).add(1);

        for (int rowNum = 1; rowNum < numRows; rowNum++) {
            List<Integer> row = new ArrayList<>();
            List<Integer> prevRow = triangle.get(rowNum-1);

            // The first row element is always 1.
            row.add(1);

            // Each triangle element (other than the first and last of each row)
            // is equal to the sum of the elements above-and-to-the-left and
            // above-and-to-the-right.
            for (int j = 1; j < rowNum; j++) {
                row.add(prevRow.get(j-1) + prevRow.get(j));
            }

            // The last row element is always 1.
            row.add(1);

            triangle.add(row);
        }

        return triangle;
    }
    /**
     * 问题4：杨辉三角
     * 具体描述如上。
     * 解决方案：递归
     *
     * @param numRows 给定行数
     * @return 杨辉三角集合
     */
    public static List<List<Integer>> generatePascalTriangleByRecursive(int numRows) {
        if (numRows < 0) return null;

        List<List<Integer>> res = new ArrayList<>();
        if (numRows < 1) return res;

        for (int i = 0; i < numRows; i++) {
            List<Integer> rowRes = new ArrayList<>();
            for (int j = 0; j < i + 1; j++) {
                rowRes.add(generateHelper(i, j));
            }
            res.add(rowRes);
        }

        return res;
    }
    public static Integer generateHelper(int i, int j) {
        if (j == 0 || j == i) {
            return 1;
        }

        return generateHelper(i - 1, j - 1) + generateHelper(i - 1, j);
    }

    /**
     * 问题5：杨辉三角2
     * 具体描述：给定一个非负索引 k，其中 k ≤ 33，返回杨辉三角的第 k 行。
     *         在杨辉三角中，每个数是它左上方和右上方的数的和。
     * 示例:
     * 输入: 3
     * 输出: [1,3,3,1]
     * 进阶：你可以优化你的算法到 O(k) 空间复杂度吗？
     *
     * @param rowIndex 给定行数
     * @return 给定行数改行集合
     */
    // 方法一：需要一层一层的求。但是不需要把每一层的结果都保存起来，只需要保存上一层的结果，就可以求出当前层的结果了。
    public List<Integer> getRowByListPre(int rowIndex) {
        List<Integer> pre = new ArrayList<>();
        List<Integer> cur = new ArrayList<>();
        for (int i = 0; i <= rowIndex; i++) {
            cur = new ArrayList<>();
            for (int j = 0; j <= i; j++) {
                if (j == 0 || j == i) {
                    cur.add(1);
                } else {
                    cur.add(pre.get(j - 1) + pre.get(j));
                }
            }
            pre = cur;
        }
        return cur;
    }
    // 方法二：方法一可以优化一下，我们可以把 pre 的 List 省去。这样的话，cur每次不去新建 List，而是把cur当作pre。
    // 又因为更新当前j的时候，就把之前j的信息覆盖掉了。而更新 j + 1 的时候又需要之前j的信息，所以在更新前，我们需要一个变量把之前j的信息保存起来。
    public List<Integer> getRowByIntPre(int rowIndex) {
        int pre = 1;
        List<Integer> cur = new ArrayList<>();
        cur.add(1);
        for (int i = 1; i <= rowIndex; i++) {
            for (int j = 1; j < i; j++) {
                int temp = cur.get(j);
                cur.set(j, pre + cur.get(j));
                pre = temp;
            }
            cur.add(1);
        }
        return cur;
    }
    // 方法三：除了上边优化的思路，还有一种想法，那就是倒着进行，这样就不会存在覆盖的情况了。
    // 因为更新完j的信息后，虽然把j之前的信息覆盖掉了。但是下一次我们更新的是j - 1，需要的是j - 1和j - 2 的信息，j信息覆盖就不会造成影响了。
    public static List<Integer> getRowByBackward(int rowIndex) {
        // 初始化集合
        List<Integer> cur = new ArrayList<>();

        // 每层的第一个 1
        cur.add(1);

        for (int i = 1; i <= rowIndex; i++) {
            for (int j = i - 1; j > 0; j--) {
                cur.set(j, cur.get(j - 1) + cur.get(j));
            }
            // 补上每层的最后一个 1
            cur.add(1);
        }
        return cur;
    }
    // 方法四：公式法
    // 如果熟悉杨辉三角，应该记得杨辉三角其实可以看做由组合数构成。
    // 公式如下：C_n^k
    // 其中：n为当前行索引，k为当前列索引
    public List<Integer> getRowByFormula(int rowIndex) {
        List<Integer> ans = new ArrayList<>();
        for (int k = 0; k <= rowIndex; k++) {
            ans.add(Combination(rowIndex, k));
        }
        return ans;
    }
    private int Combination(int n, int k) {
        long res = 1;
        for (int i = 1; i <= k; i++)
            res = res * (n - k + i) / i;
        return (int) res;
    }
    // 方法五：方法四可以优化一下。
    // 上边的算法对于每个组合数我们都重新求了一遍，但事实上前后的组合数其实是有联系的。
    //     C_n^k=C_n^{k-1} * (n-k+1) / k
    // 代码的话，我们只需要用pre变量保存上一次的组合数结果。计算过程中，可能越界，所以用到了long。
    public List<Integer> getRow(int rowIndex) {
        List<Integer> ans = new ArrayList<>();
        long pre = 1;
        ans.add(1);
        for (int k = 1; k <= rowIndex; k++) {
            long cur = pre * (rowIndex - k + 1) / k;
            ans.add((int) cur);
            pre = cur;
        }
        return ans;
    }

    /**
     * 问题6：反转链表
     * 具体描述：反转一个单链表。
     *
     * 示例:
     * 输入: 1->2->3->4->5->NULL
     * 输出: 5->4->3->2->1->NULL
     *
     * 进阶: 你可以迭代或递归地反转链表。你能否用两种方法解决这道题？
     *
     * @param head 头节点
     * @return 反转后的头节点
     */
    // 方法一：递归
    public static ListNode reverseListByRecursive(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode sonNode = reverseListByRecursive(head.next);
        head.next.next = head;
        head.next = null;

        return sonNode;
    }
    // 方法二：迭代
    public static ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while (curr != null) {
            // 保存当前节点的next
            ListNode nextTemp = curr.next;
            // 将当前节点的next置为prev
            curr.next = prev;
            // 更改prev的指向
            prev = curr;
            // 更改curr的指向
            curr = nextTemp;
        }

        return prev;
    }

    /**
     * 问题7：斐波那契数
     * 具体描述：斐波那契数，通常用 F(n) 表示，形成的序列称为斐波那契数列。该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：
     *           F(0) = 0,   F(1) = 1
     *           F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
     *         给定 N，计算 F(N)。
     * 示例 1：
     * 输入：2
     * 输出：1
     * 解释：F(2) = F(1) + F(0) = 1 + 0 = 1.
     *
     * 示例 2：
     * 输入：3
     * 输出：2
     * 解释：F(3) = F(2) + F(1) = 1 + 1 = 2.
     *
     * 示例 3：
     * 输入：4
     * 输出：3
     * 解释：F(4) = F(3) + F(2) = 2 + 1 = 3.
     *
     * 提示：
     * 0 ≤ N ≤ 30
     *
     * @param N 给定的非负整数
     * @return 斐波那契数
     */
    // 方法一：会重复计算，无缓存
    public static int fib(int N) {
        if (N < 2) return N;
        return fib(N - 1) + fib(N - 2);
    }
    // 方法二：对方法一进行优化，增加缓存技术
    static HashMap<Integer, Integer> cache = new HashMap<>();
    public static int fibWithCache(int N) {
        if (cache.containsKey(N)) return cache.get(N);

        int result = 0;
        if (N < 2) result = N;
        else result = fibWithCache(N - 1) + fibWithCache(N - 2);

        cache.put(N, result);

        return result;
    }

    /**
     * 问题8：爬楼梯
     * 具体描述：假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
     *         每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
     *         注意：给定 n 是一个正整数。
     * 示例 1：
     * 输入： 2
     * 输出： 2
     * 解释： 有两种方法可以爬到楼顶。
     * 1.  1 阶 + 1 阶
     * 2.  2 阶
     *
     * 示例 2：
     * 输入： 3
     * 输出： 3
     * 解释： 有三种方法可以爬到楼顶。
     * 1.  1 阶 + 1 阶 + 1 阶
     * 2.  1 阶 + 2 阶
     * 3.  2 阶 + 1 阶
     *
     * @param n 给定的正整数，表示楼梯阶数
     * @return 表是有多少种方法可以到达给定楼梯阶数
     */
    // 方法一：暴力法
    // 在暴力法中，我们将会把所有可能爬的阶数进行组合，也就是 1 和 2。
    // 而在每一步中我们都会继续调用 climbStairsclimbStairs 这个函数模拟爬 11 阶和 22 阶的情形，并返回两个函数的返回值之和。
    //     climbStairs(i,n) = climbStairs(i + 1, n) + climbStairs(i + 2, n)
    // 其中 ii 定义了当前阶数，而 nn 定义了目标阶数。
    //
    // 复杂度分析：
    //     时间复杂度：O(2^n)。树形递归的大小为 2^n
    //     空间复杂度：O(n)。递归树的深度可以达到 n
    public static int climbStairsWithViolence(int n) {
        return climbStairsWithViolenceHelper(0, n);
    }
    public static int climbStairsWithViolenceHelper(int i, int n) {
        if (i > n) {
            return 0;
        }
        if (i == n) {
            return 1;
        }
        return climbStairsWithViolenceHelper(i + 1, n) + climbStairsWithViolenceHelper(i + 2, n);
    }
    // 方法二：记忆化递归（暴力法的改进版本）
    // 在上一种方法中，我们计算每一步的结果时出现了冗余。
    // 另一种思路是，我们可以把每一步的结果存储在 memomemo 数组之中，每当函数再次被调用，我们就直接从 memomemo 数组返回结果。
    // 在 memomemo 数组的帮助下，我们得到了一个修复的递归树，其大小减少到 n。
    //
    // 复杂度分析：
    //     时间复杂度：O(n)。树形递归的大小可以达到 n。
    //     空间复杂度：O(n)。递归树的深度可以达到 n。
    public static int climbStairsWithMemory(int n) {
        int[] memo = new int[n + 1];
        return climbStairsWithMemoryHelper(0, n, memo);
    }
    public static int climbStairsWithMemoryHelper(int i, int n, int[] memo) {
        if (i > n) {
            return 0;
        }
        if (i == n) {
            return 1;
        }
        if (memo[i] > 0) {
            return memo[i];
        }
        memo[i] = climbStairsWithMemoryHelper(i + 1, n, memo) + climbStairsWithMemoryHelper(i + 2, n, memo);
        return memo[i];
    }
    // 方法三：动态规划
    // 不难发现，这个问题可以被分解为一些包含最优子结构的子问题，即它的最优解可以从其子问题的最优解来有效地构建，我们可以使用动态规划来解决这一问题。
    // 第 i 阶可以由以下两种方法得到：
    //     在第 i-1 阶后向上爬 1 阶。
    //     在第 i-2 阶后向上爬 2 阶。
    // 所以到达第 i 阶的方法总数就是到第 i−1 阶和第 i−2 阶的方法数之和。
    // 令 dp[i]dp[i] 表示能到达第 ii 阶的方法总数：
    //     dp[i]=dp[i-1]+dp[i-2]
    //
    // 复杂度分析：
    //     时间复杂度：O(n)，单循环到 n。
    //     空间复杂度：O(n)。dp 数组用了 n 的空间。
    public static int climbStairsWithDP(int n) {
        if (n == 1) {
            return 1;
        }
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
    // 方法 4: 斐波那契数
    // 在上述方法中，我们使用 dp 数组，其中 dp[i]=dp[i-1]+dp[i-2]。
    // 可以很容易通过分析得出 dp[i] 其实就是第 i 个斐波那契数。
    //     Fib(n)=Fib(n−1)+Fib(n−2)
    // 现在我们必须找出以 1 和 2 作为第一项和第二项的斐波那契数列中的第 n 个数，也就是说 Fib(1)=1 且 Fib(2)=2。

    // 复杂度分析：
    //    时间复杂度：O(n)。单循环到 n，需要计算第 n 个斐波那契数。
    //    空间复杂度：O(1)。使用常量级空间。
    public static int climbStairsWithFibNumber(int n) {
        if (n == 1) {
            return 1;
        }
        int first = 1;
        int second = 2;
        for (int i = 3; i <= n; i++) {
            int third = first + second;
            first = second;
            second = third;
        }
        return second;
    }
    // 方法5: 斐波那契公式
    // 对于给定的问题，斐波那契序列将会被定义为 F_0 = 1, F_1 = 1, F_2 = 2，F_{n+2}= F_{n+1} + F_n。
    // 尝试解决这一递归公式的标准方法是设出 F_n，其形式为 F_n= a^n。
    // 然后，自然有 F_{n+1} = a^{n+1} 和 F_{n+2}= a^{n+2}，所以方程可以写作 a^{n+2}= a^{n+1}+ a^n。
    // 如果我们对整个方程进行约分，可以得到 a^2= a + 1 或者写成二次方程形式 a^2 - a- 1= 0。

    // 通过解上述二元一次方程，可得如下公式来找出第 n 个斐波那契数：
    //     F(n) = [(1+sqrt(5)/2))^n - (1-sqrt(5))/2))^n] / sqrt(5)
    //
    // 复杂度分析：
    //     时间复杂度：O(log(n))。pow 方法将会用去 log(n) 的时间。
    //     空间复杂度：O(1)。使用常量级空间。
    public static int climbStairsWithFibFormula(int n) {
        double sqrt5 = Math.sqrt(5);
        double fibNum = Math.pow((1 + sqrt5) / 2, n + 1) - Math.pow((1 - sqrt5) / 2, n + 1);
        return (int)(fibNum / sqrt5);
    }
    // 方法 6: Binets 方法
    // 这里有一种有趣的解法，它使用矩阵乘法来得到第 n 个斐波那契数。矩阵形式如下：
    //     具体可上网自查：https://leetcode-cn.com/explore/orignial/card/recursion-i/258/memorization/1214/
    // 复杂度分析：
    //     时间复杂度：O(log(n))。遍历 log(n) 位。
    //     空间复杂度：O(1)。使用常量级空间。
    public static int climbStairsWithBinets(int n) {
        int[][] q = {{1, 1}, {1, 0}};
        int[][] res = powWithBinets(q, n);
        return res[0][0];
    }
    public static int[][] powWithBinets(int[][] a, int n) {
        int[][] ret = {{1, 0}, {0, 1}};
        while (n > 0) {
            if ((n & 1) == 1) {
                ret = multiplyWithBinets(ret, a);
            }
            n >>= 1;
            a = multiplyWithBinets(a, a);
        }
        return ret;
    }
    public static int[][] multiplyWithBinets(int[][] a, int[][] b) {
        int[][] c = new int[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
            }
        }
        return c;
    }

    /**
     * 问题9：二叉树的最大深度
     * 具体描述：给定一个二叉树，找出其最大深度。
     *         二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
     *         说明: 叶子节点是指没有子节点的节点。
     * 示例：
     * 给定二叉树 [3,9,20,null,null,15,7]，
     *     3
     *    / \
     *   9  20
     *     /  \
     *    15   7
     * 返回它的最大深度 3。
     *
     * @param root 根节点
     * @param <T> 类型参数
     * @return 根节点最大深度
     */
    // 方法一：递归
    // 复杂度分析：
    //     时间复杂度：我们每个结点只访问一次，因此时间复杂度为 O(N)， 其中 N 是结点的数量。
    //     空间复杂度：在最糟糕的情况下，树是完全不平衡的，例如每个结点只剩下左子结点，递归将会被调用 N 次（树的高度），因此保持调用栈的存储将是 O(N)。
    //              但在最好的情况下（树是完全平衡的），树的高度将是 log(N)。因此，在这种情况下的空间复杂度将是 O(log(N))。
    public static <T> int maxDepthWithRecursive(BinaryNode<T> root) {
        if (root == null) {
            return 0;
        }

        int left_depth = maxDepthWithRecursive(root.getLeft());
        int right_depth = maxDepthWithRecursive(root.getRight());

        return Math.max(left_depth, right_depth) + 1;
    }
    // 方法二：迭代
    // 我们还可以在栈的帮助下将上面的递归转换为迭代。
    // 具体方法：使用 DFS 策略访问每个结点，同时在每次访问时更新最大深度。
    // 所以我们从包含根结点且相应深度为 1 的栈开始。然后我们继续迭代：将当前结点弹出栈并推入子结点。每一步都会更新深度。
    //
    // 复杂度分析：
    //     时间复杂度：O(N).
    //     空间复杂度：O(N).
    public static <T> int maxDepthWithIteration(BinaryNode<T> root) {
        Queue<Pair<BinaryNode<T>, Integer>> stack = new LinkedList<>();
        if (root != null) {
            stack.add(new Pair(root, 1));
        }

        int depth = 0;
        while (!stack.isEmpty()) {
            Pair<BinaryNode<T>, Integer> current = stack.poll();
            root = current.getKey();
            int current_depth = current.getValue();
            if (root != null) {
                depth = Math.max(depth, current_depth);
                stack.add(new Pair(root.getLeft(), current_depth + 1));
                stack.add(new Pair(root.getRight(), current_depth + 1));
            }
        }
        return depth;
    }

    /**
     * 问题10：计算 x 的 n 次幂函数
     * 具体描述：实现 pow(x, n) ，即计算 x 的 n 次幂函数。
     * 示例 1:
     * 输入: 2.00000, 10
     * 输出: 1024.00000
     *
     * 示例 2:
     * 输入: 2.10000, 3
     * 输出: 9.26100
     *
     * 示例 3:
     * 输入: 2.00000, -2
     * 输出: 0.25000
     * 解释: 2-2 = 1/22 = 1/4 = 0.25
     *
     * 说明:
     *     -100.0 < x < 100.0
     *     n 是 32 位有符号整数，其数值范围是 [−231, 231 − 1] 。
     * @param x 基数
     * @param n 次方
     * @return x 的 n 次方
     */
    // 方法一：暴力法
    // 思路：只需模拟将 x 相乘 n 次的过程。
    //      如果 n &lt; 0n<0，我们可以直接用 1 / x, −n 来替换 x, n 以保证 n ≥ 0。该限制可以简化我们的进一步讨论。
    //      但我们需要注意极端情况，尤其是负整数和正整数的不同范围限制。
    // 我们可以用一个简单的循环来计算结果。
    public static double myPowWithViolence(double x, int n) {
        long N = n;
        if (N < 0) {
            x = 1 / x;
            N = -N;
        }
        double ans = 1;
        for (long i = 0; i < N; i++)
            ans = ans * x;
        return ans;
    }
    // 方法二：递归快速幂算法
    // 假设我们已经得到了 x ^ n 的结果，那么如何才能算出 x ^ {2 * n } ？
    // 显然，我们不需要再乘上 n 个 x。 使用公式 (x ^ n) ^ 2 = x ^ {2 * n }，只需要一次计算,我们就可以得到 x ^ {2 * n }。
    // 这种优化可以降低算法的时间复杂度。

    // 复杂度分析：
    //     时间复杂度：O(log(n))。每次我们应用公式 (x ^ n) ^ 2 = x ^ {2 * n}，n 就减少一半。
    //              因此，我们最多需要 O(log(n)) 次计算来得到结果。
    //     空间复杂度：O(log(n))。每次计算，我们都需要存储 x ^ {n / 2} 的结果。
    //              我们需要计算 O(log(n)) 次，因此空间复杂度为 O(log(n))。
    private static double recursiveFastPower(double x, long n) {
        if (n == 0) {
            return 1.0;
        }
        double half = recursiveFastPower(x, n / 2);
        if (n % 2 == 0) {
            return half * half;
        } else {
            return half * half * x;
        }
    }
    public static double myPowWithRecursiveFastPower(double x, int n) {
        long N = n;
        if (N < 0) {
            x = 1 / x;
            N = -N;
        }

        return recursiveFastPower(x, N);
    }
    // 方法三：迭代快速幂算法
    // 使用公式 x ^ {a + b} = x ^ a * x ^ b，我们可以将 n 写成一些正整数的和，n = ∑(i)B(i)。
    // 如果我们可以快速得到 x ^ b(i) 的结果，那么就可以减少计算 x ^ n所需的时间。
    //
    // 复杂度分析：
    //     时间复杂度：O(log(n))。对于 n 的每个二进制位，我们最多只能乘一次。所以总的时间复杂度为 O(log(n))。
    //     空间复杂度：O(1)。我们只需要两个变量来存储 x 的当前乘积和最终结果。
    public static double myPowWithIterativeFastPower(double x, int n) {
        long N = n;
        if (N < 0) {
            x = 1 / x;
            N = -N;
        }
        double ans = 1;
        double current_product = x;
        for (long i = N; i > 0; i /= 2) {
            if ((i % 2) == 1) {
                ans = ans * current_product;
            }
            current_product = current_product * current_product;
        }
        return ans;
    }

    /**
     * 问题11：合并两个有序链表
     * 具体描述：将两个升序链表合并为一个新的升序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
     * 示例：
     * 输入：1->2->4, 1->3->4
     * 输出：1->1->2->3->4->4
     *
     * @param l1 前一个链表
     * @param l2 后一个链表
     * @return 最终的有序链表
     */
    // 方法一：递归
    // 我们直接将以上递归过程建模，首先考虑边界情况。
    // 特殊的，如果 l1 或者 l2 一开始就是 null ，那么没有任何操作需要合并，所以我们只需要返回非空链表。
    // 否则，我们要判断 l1 和 l2 哪一个的头元素更小，然后递归地决定下一个添加到结果里的值。
    // 如果两个链表都是空的，那么过程终止，所以递归过程最终一定会终止。
    //
    // 复杂度分析：
    //     时间复杂度：O(n + m)。
    //         因为每次调用递归都会去掉 l1 或者 l2 的头元素（直到至少有一个链表为空），函数 mergeTwoList 中只会遍历每个元素一次。
    //         所以，时间复杂度与合并后的链表长度为线性关系。
    //     空间复杂度：O(n + m)。
    //         调用 mergeTwoLists 退出时 l1 和 l2 中每个元素都一定已经被遍历过了，所以 n + m 个栈帧会消耗 O(n + m) 的空间。
    public static ListNode mergeTwoListsByRecursive(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        }
        else if (l2 == null) {
            return l1;
        }
        else if (l1.val < l2.val) {
            l1.next = mergeTwoListsByRecursive(l1.next, l2);
            return l1;
        }
        else {
            l2.next = mergeTwoListsByRecursive(l1, l2.next);
            return l2;
        }
    }
    // 方法 2：迭代
    // 我们可以用迭代的方法来实现上述算法。我们假设 l1 元素严格比 l2元素少，我们可以将 l2 中的元素逐一插入 l1 中正确的位置。
    //
    // 复杂度分析：
    //     时间复杂度：O(n + m)。
    //         因为每次循环迭代中，l1 和 l2 只有一个元素会被放进合并链表中， while 循环的次数等于两个链表的总长度。
    //         所有其他工作都是常数级别的，所以总的时间复杂度是线性的。
    //     空间复杂度：O(1)。迭代的过程只会产生几个指针，所以它所需要的空间是常数级别的。
    public static ListNode mergeTwoListsByIteration(ListNode l1, ListNode l2) {
        // maintain an unchanging reference to node ahead of the return node.
        ListNode prehead = new ListNode(-1);

        ListNode prev = prehead;
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                prev.next = l1;
                l1 = l1.next;
            } else {
                prev.next = l2;
                l2 = l2.next;
            }
            prev = prev.next;
        }

        // exactly one of l1 and l2 can be non-null at this point, so connect
        // the non-null list to the end of the merged list.
        prev.next = l1 == null ? l2 : l1;

        return prehead.next;
    }
}
