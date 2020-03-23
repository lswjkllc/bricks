package solution;

import org.omg.PortableInterceptor.INACTIVE;

import java.util.*;

public class MyStack {

    /**
     * 问题1：有效的括号
     * 具体描述：给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
     *         有效字符串需满足：
     *             左括号必须用相同类型的右括号闭合。
     *             左括号必须以正确的顺序闭合。
     *             注意空字符串可被认为是有效字符串。
     *
     * 示例 1:
     * 输入: "()"
     * 输出: true
     *
     * 示例 2:
     * 输入: "()[]{}"
     * 输出: true
     *
     * 示例 3:
     * 输入: "(]"
     * 输出: false
     *
     * 示例 4:
     * 输入: "([)]"
     * 输出: false
     *
     * 示例 5:
     * 输入: "{[]}"
     * 输出: true
     * @param s 只包含 '('，')'，'{'，'}'，'['，']' 的字符串
     * @return 是否为有效的括号字符串
     */
    public static boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(' || c == '[' || c == '{') {
                stack.push(c);
            }
            else {
                if (stack.isEmpty()) return false;

                char beforeChar = stack.pop();
                if (c == ')' && beforeChar != '(') return false;
                if (c == ']' && beforeChar != '[') return false;
                if (c == '}' && beforeChar != '{') return false;
            }
        }
        return stack.isEmpty();
    }

    /**
     * 问题2：每日温度
     * 具体描述：根据每日 气温 列表，请重新生成一个列表，对应位置的输入是你需要再等待多久温度才会升高超过该日的天数。如果之后都不会升高，请在该位置用 0 来代替。
     *         例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。
     *         提示：气温 列表长度的范围是 [1, 30000]。每个气温的值的均为华氏度，都是在 [30, 100] 范围内的整数。
     * @param T 温度数组
     * @return 温度增长数组
     */
    public int[] dailyTemperatures(int[] T) {
        Stack<Integer> s = new Stack<Integer>();
        int[] res = new int[T.length];
        for (int i = 0; i < T.length; i++) {
            // 写成循环是针对一天的温度可能同时匹配前面多天的
            while(!s.empty() && T[i] > T[s.peek()]) {
                // 一有较大的温度就出栈配对
                res[s.peek()] = i - s.pop();
            }
            // 因为要计算该处的下一个温度上升点，所以要入栈
            s.push(i);
        }

        return res;
    }

    /**
     * 问题3：逆波兰表达式求值
     * 具体描述：根据逆波兰表示法，求表达式的值。
     *         有效的运算符包括 +, -, *, / 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。
     *         说明：
     *             整数除法只保留整数部分。
     *             给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。
     * 示例 1：
     * 输入: ["2", "1", "+", "3", "*"]
     * 输出: 9
     * 解释: ((2 + 1) * 3) = 9
     *
     * 示例 2：
     * 输入: ["4", "13", "5", "/", "+"]
     * 输出: 6
     * 解释: (4 + (13 / 5)) = 6
     *
     * 示例 3：
     * 输入: ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
     * 输出: 22
     * 解释:
     *   ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
     * = ((10 * (6 / (12 * -11))) + 17) + 5
     * = ((10 * (6 / -132)) + 17) + 5
     * = ((10 * 0) + 17) + 5
     * = (0 + 17) + 5
     * = 17 + 5
     * = 22
     * @param tokens 输入字符串数组
     * @return 最终计算值
     */
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        for (String curVal : tokens) {
            if ("+".equals(curVal) || "-".equals(curVal) || "*".equals(curVal) || "/".equals(curVal)) {
                int f = stack.pop();
                int s = stack.pop();
                int fs = 0;
                switch (curVal) {
                    case "+":
                        fs = f + s;
                        break;
                    case "-":
                        fs = f - s;
                        break;
                    case "*":
                        fs = f * s;
                        break;
                    case "/":
                        fs = f / s;
                        break;
                }
                stack.push(fs);
            } else {
                stack.push(Integer.valueOf(curVal));
            }
        }

        return stack.pop();
    }

    /**
     * 问题4：岛屿数量
     * 具体描述：给定一个由 '1'（陆地）和 '0'（水）组成的的二维网格，计算岛屿的数量。
     *         一个岛被水包围，并且它是通过水平方向或垂直方向上相邻的陆地连接而成的。
     *         你可以假设网格的四个边均被水包围。
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
     * 使用"深度优先搜索"处理岛屿数量
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
                    dfs(grid, r, c);
                }
            }
        }

        return num_islands;
    }
    /**
     * "深度优先搜索"处理岛屿数量
     * @param grid 数字网格
     * @param r 行索引
     * @param c 列索引
     */
    public static void dfs(char[][] grid, int r, int c) {
        int nr = grid.length;
        int nc = grid[0].length;

        if (r < 0 || c < 0 || r >= nr || c >= nc || grid[r][c] == '0') {
            return ;
        }

        grid[r][c] = '0';
        dfs(grid, r - 1, c);
        dfs(grid, r + 1, c);
        dfs(grid, r, c - 1);
        dfs(grid, r, c + 1);
    }

    int count = 0;
    /**
     * 问题5：目标和
     * 具体描述：给定一个非负整数数组，a1, a2, ..., an, 和一个目标数，S。现在你有两个符号 + 和 -。
     *         对于数组中的任意一个整数，你都可以从 + 或 -中选择一个符号添加在前面。
     *         返回可以使最终数组和为目标数 S 的所有添加符号的方法数。
     * 示例 1:
     * 输入: nums: [1, 1, 1, 1, 1], S: 3
     * 输出: 5
     * 解释:
     * -1+1+1+1+1 = 3
     * +1-1+1+1+1 = 3
     * +1+1-1+1+1 = 3
     * +1+1+1-1+1 = 3
     * +1+1+1+1-1 = 3
     * 一共有5种方法让最终目标和为3。
     *
     * 注意:
     * 数组非空，且长度不会超过20。
     * 初始的数组的和不会超过1000。
     * 保证返回的最终结果能被32位整数存下。
     *
     * @param nums 非负整数数组
     * @param S 目标数
     * @return 达到目标数的方法个数
     */
    public int findTargetSumWays(int[] nums, int S) {
        calculate(nums, 0, 0, S);
        return count;
    }
    public void calculate(int[] nums, int i, int sum, int S) {
        if (i == nums.length) {
            if (sum == S)
                count++;
        } else {
            calculate(nums, i + 1, sum + nums[i], S);
            calculate(nums, i + 1, sum - nums[i], S);
        }
    }

    /**
     * 问题6：字符串解码
     * 具体描述：给定一个经过编码的字符串，返回它解码后的字符串。
     *         编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。
     *         你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。
     *         此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。
     * 示例:
     * s = "3[a]2[bc]", 返回 "aaabcbc".
     * s = "3[a2[c]]", 返回 "accaccacc".
     * s = "2[abc]3[cd]ef", 返回 "abcabccdcdcdef".
     * s = "10[leet]", 返回 "leetleetleetleetleetleetleetleetleetleet".
     *
     * 算法流程：
     * 1、构建辅助栈 stack， 遍历字符串 s 中每个字符 c；
     *     当 c 为数字时，将数字字符转化为数字 multi，用于后续倍数计算；
     *     当 c 为字母时，在 res 尾部添加 c；
     *     当 c 为 [ 时，将当前 multi 和 res 入栈，并分别置空置 00：
     *         记录此 [ 前的临时结果 res 至栈，用于发现对应 ] 后的拼接操作；
     *         记录此 [ 前的倍数 multi 至栈，用于发现对应 ] 后，获取 multi × [...] 字符串。
     *         进入到新 [ 后，res 和 multi 重新记录。
     *     当 c 为 ] 时，stack 出栈，拼接字符串 res = last_res + cur_multi * res，其中:
     *         last_res是上个 [ 到当前 [ 的字符串，例如 "3[a2[c]]" 中的 a；
     *         cur_multi是当前 [ 到 ] 内字符串的重复倍数，例如 "3[a2[c]]" 中的 2。
     * 2、返回字符串 res。
     *
     * @param s 字符串
     * @return 解码后的字符串
     */
    public static String decodeString(String s) {
        StringBuilder res = new StringBuilder();
        int multi = 0;
        Stack<Integer> stack_multi = new Stack<>();
        Stack<String> stack_res = new Stack<>();

        for(Character c : s.toCharArray()) {
            if(c == '[') {
                stack_multi.push(multi);
                stack_res.push(res.toString());
                multi = 0;
                res = new StringBuilder();
            }
            else if(c == ']') {
                StringBuilder tmp = new StringBuilder();
                int cur_multi = stack_multi.pop();
                for(int i = 0; i < cur_multi; i++) tmp.append(res);
                res = new StringBuilder(stack_res.pop() + tmp);
            }
            else if(c >= '0' && c <= '9') multi = multi * 10 + Integer.parseInt(c + "");
            else res.append(c);
        }
        return res.toString();
    }

    /**
     * 问题7：图像渲染
     * 具体描述：有一幅以二维整数数组表示的图画，每一个整数表示该图画的像素值大小，数值在 0 到 65535 之间。
     *         给你一个坐标 (sr, sc) 表示图像渲染开始的像素值（行 ，列）和一个新的颜色值 newColor，让你重新上色这幅图像。
     *         为了完成上色工作，从初始坐标开始，记录初始坐标的上下左右四个方向上像素值与初始坐标相同的相连像素点，
     *         接着再记录这四个方向上符合条件的像素点与他们对应四个方向上像素值与初始坐标相同的相连像素点，……，重复该过程。将所有有记录的像素点的颜色值改为新的颜色值。
     *         最后返回经过上色渲染后的图像。
     * 示例 1:
     * 输入:
     * image = [[1,1,1],[1,1,0],[1,0,1]]
     * sr = 1, sc = 1, newColor = 2
     * 输出: [[2,2,2],[2,2,0],[2,0,1]]
     * 解析:
     * 在图像的正中间，(坐标(sr,sc)=(1,1)),
     * 在路径上所有符合条件的像素点的颜色都被更改成2。
     * 注意，右下角的像素没有更改为2，
     * 因为它不是在上下左右四个方向上与初始点相连的像素点。
     *
     * 注意:
     * image 和 image[0] 的长度在范围 [1, 50] 内。
     * 给出的初始点将满足 0 <= sr < image.length 和 0 <= sc < image[0].length。
     * image[i][j] 和 newColor 表示的颜色值在范围 [0, 65535]内。
     *
     * @param image 由图像的像素组成的二维数组
     * @param sr 指定行
     * @param sc 指定列
     * @param newColor 新值
     * @return 图像渲染后的像素组成的二维数组
     */
    public static int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        if (image == null) return null;
        int[][] visited = new int[image.length][image[0].length];
        changeColor(image, sr, sc, image[sr][sc], newColor, visited);

        return image;
    }
    public static void changeColor(int[][] image, int sr, int sc, int oldColor, int newColor, int[][] visited) {
        if (sr >= 0 && sr < image.length && sc >=0 && sc < image[0].length && visited[sr][sc] == 0 && image[sr][sc] == oldColor) {
            visited[sr][sc] = 1;
            image[sr][sc] = newColor;

            changeColor(image, sr - 1, sc, oldColor, newColor, visited);
            changeColor(image, sr + 1, sc, oldColor, newColor, visited);
            changeColor(image, sr, sc - 1, oldColor, newColor, visited);
            changeColor(image, sr, sc + 1, oldColor, newColor, visited);
        }
    }

    /**
     * 问题8：01矩阵
     * 具体描述：给定一个由 0 和 1 组成的矩阵，找出每个元素到最近的 0 的距离。
     *         两个相邻元素间的距离为 1 。
     * 示例 1:
     * 输入:
     * 0 0 0
     * 0 1 0
     * 0 0 0
     * 输出:
     * 0 0 0
     * 0 1 0
     * 0 0 0
     *
     * 示例 2:
     * 输入:
     * 0 0 0
     * 0 1 0
     * 1 1 1
     * 输出:
     * 0 0 0
     * 0 1 0
     * 1 2 1
     *
     * 注意:
     * 给定矩阵的元素个数不超过 10000。
     * 给定矩阵中至少有一个元素是 0。
     * 矩阵中的元素只在四个方向上相邻: 上、下、左、右。
     *
     * 解题思路：最开始的时候，我的思路是遍历所有的1，对于每个1，一层一层向外搜索，直至找到0，这样超时了。看了大佬们的讲解，思路豁然开朗.
     *         1、首先遍历matrix，对于非零点，设置一个较大值（row+col）
     *         2、维护一个队列，首先将所有零点的坐标放入队列中
     *         3、取出队列中的元素（i，j），搜索（i，j）的四个方向，如果某方向上的值大于或等于（matrix[i][j]+1),
     *            就将该方向的坐标值更新为matrix[i][j]+1,这是局部正确的。
     *         4、然后将该方向的坐标加入队列
     *         5、重复3-4步骤，直到队列为空。
     *
     * @param matrix 给定矩阵
     * @return 返回距离1最近0矩阵
     */
    public static int[][] updateMatrix(int[][] matrix) {
        int row = matrix.length;
        int col = matrix[0].length;
        //灵活应对四个方向的变化
        final int[][] vector = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (matrix[i][j] == 0) {
                    // 将所有 0 元素作为 BFS 第一层
                    queue.add(new int[]{i, j});
                } else {
                    //设一个最大值
                    matrix[i][j] = row + col;
                }
            }
        }
        while (!queue.isEmpty()) {
            int[] s = queue.poll();
            // 搜索上下左右四个方向
            for (int[] v : vector) {
                int r = s[0] + v[0];
                int c = s[1] + v[1];
                if (r >= 0 && r < row && c >= 0 && c < col){
                    if (matrix[r][c] >= matrix[s[0]][s[1]] + 1){
                        matrix[r][c] = matrix[s[0]][s[1]] + 1;
                        queue.add(new int[]{r, c});
                    }
                }
            }
        }
        return matrix;
    }

    /**
     * 问题9：钥匙和房间
     * 具体描述：有 N 个房间，开始时你位于 0 号房间。每个房间有不同的号码：0，1，2，...，N-1，并且房间里可能有一些钥匙能使你进入下一个房间。
     *         在形式上，对于每个房间 i 都有一个钥匙列表 rooms[i]，每个钥匙 rooms[i][j] 由 [0,1，...，N-1] 中的一个整数表示，
     *             其中 N = rooms.length。 钥匙 rooms[i][j] = v 可以打开编号为 v 的房间。
     *         最初，除 0 号房间外的其余所有房间都被锁住。
     *         你可以自由地在房间之间来回走动。
     *         如果能进入每个房间返回 true，否则返回 false。
     * 示例 1：
     * 输入: [[1],[2],[3],[]]
     * 输出: true
     * 解释:
     * 我们从 0 号房间开始，拿到钥匙 1。
     * 之后我们去 1 号房间，拿到钥匙 2。
     * 然后我们去 2 号房间，拿到钥匙 3。
     * 最后我们去了 3 号房间。
     * 由于我们能够进入每个房间，我们返回 true。
     *
     * 示例 2：
     * 输入：[[1,3],[3,0,1],[2],[0]]
     * 输出：false
     * 解释：我们不能进入 2 号房间。
     *
     * 提示：
     * 1 <= rooms.length <= 1000
     * 0 <= rooms[i].length <= 1000
     * 所有房间中的钥匙数量总计不超过 3000。
     *
     * @param rooms 房间和钥匙组成的二维数组
     * @return 是否能进入每个房间
     */
    public static boolean canVisitAllRooms(List<List<Integer>> rooms) {

        boolean[] seen = new boolean[rooms.size()];
        seen[0] = true;

        Stack<Integer> stack = new Stack<>();
        stack.push(0);

        while (!stack.isEmpty()) { // While we have keys...
            int node = stack.pop(); // Get the next key 'node'
            for (int nei: rooms.get(node)) // For every key in room # 'node'...
                if (!seen[nei]) { // ...that hasn't been used yet
                    seen[nei] = true; // mark that we've entered the room
                    stack.push(nei); // add the key to the
                }
        }

        for (boolean v: seen)  // if any room hasn't been visited, return false
            if (!v) return false;
        return true;
    }

    public static void main(String[] args) {

    }
}

/**
 * 实现最小栈
 * 方法1：辅助栈
 */
class MinStackByTwoStack {
    private Stack<Integer> data;
    private Stack<Integer> minValues;

    /** initialize your data structure here. */
    public MinStackByTwoStack() {
        this.data = new Stack<>();
        this.minValues = new Stack<>();
    }

    public void push(int x) {
        this.data.push(x);
        if (this.minValues.isEmpty()) {
            this.minValues.push(x);
        }
        else {
            Integer topValue = this.minValues.peek();
            int minValue = topValue <= x ? topValue : x;
            this.minValues.push(minValue);
        }
    }

    public void pop() {
        if (this.data.isEmpty()) return ;
        this.data.pop();
        this.minValues.pop();
    }

    public int top() {
        return this.data.peek();
    }

    public int getMin() {
        return this.minValues.peek();
    }
}

/**
 * 实现最小栈
 * 方法2：向里面压两次的方法
 */
class MinStackByOneStack {
    private int minValue;
    private Stack<Integer> s;

    public MinStackByOneStack() {
        this.minValue = Integer.MAX_VALUE;
        s = new Stack<>();
    }

    public void push(int x) {
        if (x <= minValue) {
            s.push(minValue);
            minValue = x;
        }
        s.push(x);
    }

    public void pop() {
        if (s.pop() == minValue) minValue = s.pop();
    }

    public int top() {
        return s.peek();
    }

    public int getMin() {
        return minValue;
    }
}

/**
 * 用栈实现队列
 */
class MyQueueByStack {
    private Stack<Integer> s1;
    private Stack<Integer> s2;
    private int front;

    /** Initialize your data structure here. */
    public MyQueueByStack() {
        this.s1 = new Stack<>();
        this.s2 = new Stack<>();
    }

    /** Push element x to the back of queue. */
    public void push(int x) {
        if (this.s1.empty())
            this.front = x;
        while (!this.s1.isEmpty())
            this.s2.push(this.s1.pop());
        this.s2.push(x);
        while (!this.s2.isEmpty())
            this.s1.push(this.s2.pop());
    }

    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
        Integer res = this.s1.pop();
        if (!this.s1.empty())
            this.front = this.s1.peek();
        return res;
    }

    /** Get the front element. */
    public int peek() {
        return this.front;
    }

    /** Returns whether the queue is empty. */
    public boolean empty() {
        return this.s1.isEmpty();
    }
}

/**
 * 用队列实现栈
 */
class MyStackByQueue {
    private Queue<Integer> q1;
    private Queue<Integer> q2;
    private int front;

    /** Initialize your data structure here. */
    public MyStackByQueue() {
        this.q1 = new LinkedList<>();
        this.q2 = new LinkedList<>();
    }

    /** Push element x onto stack. */
    public void push(int x) {
        this.q1.add(x);
        this.front = x;
    }

    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
        while (this.q1.size() > 1) {
            this.front = this.q1.remove();
            this.q2.add(front);
        }
        Integer res = this.q1.remove();
        Queue<Integer> temp = this.q1;
        this.q1 = this.q2;
        this.q2 = temp;

        return res;
    }

    /** Get the top element. */
    public int top() {
        return this.front;
    }

    /** Returns whether the stack is empty. */
    public boolean empty() {
        return this.q1.isEmpty();
    }
}
