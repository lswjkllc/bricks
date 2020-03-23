package solution;

import java.util.*;

public class MyQueue {

    /**
     * 问题1：完全平方数
     * 具体描述：给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。
     *
     * 示例 1:
     * 输入: n = 12
     * 输出: 3
     * 解释: 12 = 4 + 4 + 4.
     *
     * 示例 2:
     * 输入: n = 13
     * 输出: 2
     * 解释: 13 = 4 + 9.
     *
     * 方法1："动态规划"方法求解
     * @param n 给定正整数
     * @return 需要的平方个数
     */
    public static int numSquaresByDp(int n) {
        // 默认初始化值都为0
        int[] dp = new int[n + 1];

        for (int i = 1; i <= n; i++) {
            // 最坏的情况就是每次+1
            dp[i] = i;
            for (int j = 1; j * j <= i; j++) {
                // 动态转移方程
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }
        }
        return dp[n];
    }

    /**
     * 问题1：完全平方数（见上）
     * 方法2："广度优先搜索"方法求解
     * @param n 给定正整数
     * @return 需要的平方个数
     */
    public static int numSquaresByBFS(int n) {
        // 队列
        java.util.Queue<Integer> queue = new LinkedList<>();
        queue.add(0);
        // 增加哈希列表，防止重复访问
        Set<Integer> visited = new HashSet<>();
        visited.add(0);

        int distance = 0;
        while (!queue.isEmpty()) {
            distance++;
            // 计算当前队列的长度
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                // 弹出第一个元素
                int curr = queue.remove();
                // 动态增加
                for (int j = 1; j * j + curr <= n; j++) {
                    // next = 当前值+j的平方
                    int next = j * j + curr;
                    // 如果 next=n，直接返回distance
                    if (next == n) return distance;
                    // 如果 next<n && next没有被访问过，将next加入队列和被访问列表
                    if (next < n && !visited.contains(next)) {
                        queue.add(next);
                        visited.add(next);
                    }
                }
            }
        }

        return distance;
    }

    /**
     * 问题2：打开转盘锁
     * 具体描述：你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 。每个拨轮可以自由旋转：例如把 '9' 变为  '0'，'0' 变为 '9' 。每次旋转都只能旋转一个拨轮的一位数字。
     *         锁的初始数字为 '0000' ，一个代表四个拨轮的数字的字符串。
     *         列表 deadends 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。
     *         字符串 target 代表可以解锁的数字，你需要给出最小的旋转次数，如果无论如何不能解锁，返回 -1。
     *
     * 示例1:
     * 输入：deadends = ["0201","0101","0102","1212","2002"], target = "0202"
     * 输出：6
     * 解释：可能的移动序列为 "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202"。
     *      注意 "0000" -> "0001" -> "0002" -> "0102" -> "0202" 这样的序列是不能解锁的，
     *      因为当拨动到 "0102" 时这个锁就会被锁定。
     *
     * 示例2:
     * 输入: deadends = ["0000"], target = "8888"
     * 输出：-1
     *
     * @param deadends
     * @param target
     * @return
     */
    public int openLock(String[] deadends, String target) {
        // 死亡数字转为哈希表
        HashSet<String> dead_set = new HashSet<>(Arrays.asList(deadends));

        // 死亡数字如果含有初始节点，返回-1
        if (dead_set.contains("0000")) return -1;

        // 初始化队列
        java.util.Queue<String> queue = new LinkedList<>();
        // 加入初始节点
        queue.add("0000");

        // 记录步数
        int count = 0;

        // 节点未访问完，队列内的节点不为空
        while (!queue.isEmpty()) {
            // 每一步节点数
            int size = queue.size();
            while (size-- > 0) {
                // 弹出头节点
                String tmp = queue.remove();
                //如果与目标数相同，直接返回步数
                if (target.equals(tmp)) return count;

                // 将字符串转为数组
                char[] c = tmp.toCharArray();
                // 每次修改四位数字的一位
                for (int j = 0; j < 4; j++) {
                    // char转为int型
                    int i = c[j] - '0';

                    // 数字-1。余数运算可防止节点为0、9时出现-1、10的情况
                    c[j] = (char) ('0' + (i + 9) % 10);
                    // 得到新字符串
                    String s = new String(c);
                    //字符串不在死亡数字中时
                    if (!dead_set.contains(s)) {
                        // 添加到队列作为下一步需要遍历的节点
                        queue.add(s);
                        // 下一步必访问该节点，所以可先加入到死亡数字
                        dead_set.add(s);
                    }

                    //数字+1
                    c[j] = (char) ('0' + (i + 11) % 10);
                    s = new String(c);
                    if (!dead_set.contains(s)) {
                        queue.add(s);
                        dead_set.add(s);
                    }
                    c[j] = (char) ('0' + i);
                }
            }
            count++;
        }
        return -1;
    }

    /**
     * 问题3：岛屿数量
     * 具体描述：给定一个由 '1'（陆地）和 '0'（水）组成的的二维网格，计算岛屿的数量。一个岛被水包围，
     *         并且它是通过水平方向或垂直方向上相邻的陆地连接而成的。你可以假设网格的四个边均被水包围。
     *
     * 示例 1:
     * 输入:
     * 11110
     * 11010
     * 11000
     * 00000
     * 输出: 1
     *
     * 示例 2:
     * 输入:
     * 11000
     * 11000
     * 00100
     * 00011
     * 输出: 3
     *
     * 使用"广度优先搜索"处理岛屿数量
     * @param grid 表示"岛屿"和"海"的数字网格
     * @return 岛屿数量
     */
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }

        int nr = grid.length;
        int nc = grid[0].length;
        int num_islands = 0;

        for (int r = 0; r < nr; ++r) {
            for (int c = 0; c < nc; ++c) {
                if (grid[r][c] == '1') {
                    ++num_islands;
                    grid[r][c] = '0'; // mark as visited
                    Queue<Integer> neiors = new LinkedList<>();
                    neiors.add(r * nc + c);
                    while (!neiors.isEmpty()) {
                        int id = neiors.remove();
                        int row = id / nc;
                        int col = id % nc;
                        if (row - 1 >= 0 && grid[row-1][col] == '1') {
                            neiors.add((row-1) * nc + col);
                            grid[row-1][col] = '0';
                        }
                        if (row + 1 < nr && grid[row+1][col] == '1') {
                            neiors.add((row+1) * nc + col);
                            grid[row+1][col] = '0';
                        }
                        if (col - 1 >= 0 && grid[row][col-1] == '1') {
                            neiors.add(row * nc + col-1);
                            grid[row][col-1] = '0';
                        }
                        if (col + 1 < nc && grid[row][col+1] == '1') {
                            neiors.add(row * nc + col+1);
                            grid[row][col+1] = '0';
                        }
                    }
                }
            }
        }

        return num_islands;
    }

    /**
     * 问题4：墙与门
     * 具体描述：你被给定一个 m × n 的二维网格，网格中有以下三种可能的初始化值：
     *         -1 表示墙或是障碍物
     *         0 表示一扇门
     *         INF 无限表示一个空的房间。然后，我们用 231 - 1 = 2147483647 代表 INF。你可以认为通往门的距离总是小于 2147483647 的。
     *         你要给每个空房间位上填上该房间到 最近 门的距离，如果无法到达门，则填 INF 即可。
     *
     * 示例：
     *
     * 给定二维网格：
     * INF  -1  0  INF
     * INF INF INF  -1
     * INF  -1 INF  -1
     *   0  -1 INF INF
     * 运行完你的函数后，该网格应该变成：
     *   3  -1   0   1
     *   2   2   1  -1
     *   1  -1   2  -1
     *   0  -1   3   4
     * @param rooms
     */
    public static void wallsAndGates(int[][] rooms) {
        if (rooms.length == 0) return ;

        for (int i = 0; i < rooms.length; i++) {
            for (int j = 0; j < rooms[i].length; j++) {
                if (rooms[i][j] == 0) wallsAndGatesBfs(rooms, i, j);
            }
        }
    }

    public static void wallsAndGatesBfs(int[][] rooms, int i, int j) {
        java.util.Queue<Integer> queue = new LinkedList<Integer>();
        queue.offer(i * rooms[i].length + j);
        int dist = 0;
        // 用一个集合记录已经访问过的点
        Set<Integer> visited = new HashSet<Integer>();
        visited.add(i * rooms[0].length + j);
        while(!queue.isEmpty()){
            int size = queue.size();
            // 记录深度的搜索
            for(int k = 0; k < size; k++){
                Integer curr = queue.poll();
                int row = curr / rooms[i].length;
                int col = curr % rooms[i].length;
                // 选取之前标记的值和当前的距离的较小值
                rooms[row][col] = Math.min(rooms[row][col], dist);
                int up = (row - 1) * rooms[i].length + col;
                int down = (row + 1) * rooms[i].length + col;
                int left = row * rooms[i].length + col - 1;
                int right = row * rooms[i].length + col + 1;
                if(row > 0 && rooms[row - 1][col] > 0 && !visited.contains(up)){
                    queue.offer(up);
                    visited.add(up);
                }
                if(col > 0 && rooms[row][col - 1] > 0 && !visited.contains(left)){
                    queue.offer(left);
                    visited.add(left);
                }
                if(row < rooms.length - 1 && rooms[row + 1][col] > 0 && !visited.contains(down)){
                    queue.offer(down);
                    visited.add(down);
                }
                if(col < rooms[0].length - 1 && rooms[row][col + 1] > 0 && !visited.contains(right)){
                    queue.offer(right);
                    visited.add(right);
                }
            }
            dist++;
        }
    }
}
