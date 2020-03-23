package tree;

import com.sun.tools.classfile.ConstantPool;

import java.util.*;

public class NodeUtil {

    /**
     * 前序遍历 BinaryNode
     * @param root 根节点
     * @return 前序遍历的有序节点集合
     */
    public static <T> List<T> preorderTraversal(BinaryNode<T> root) {
        List<T> vals = new ArrayList<>();
        if (root == null) {
            return vals;
        }
        vals.add(root.getValue());

        List<T> left_vals = preorderTraversal(root.getLeft());
        if (left_vals.size() != 0) { vals.addAll(left_vals); }

        List<T> right_vals = preorderTraversal(root.getRight());
        if (right_vals.size() != 0) { vals.addAll(right_vals); }

        return vals;
    }

    /**
     * 中序遍历 BinaryNode
     * @param root 根节点
     * @return 中序遍历的有序节点集合
     */
    public static <T> List<T> inorderTraversal(BinaryNode<T> root) {
        List<T> vals = new ArrayList<>();
        if (root == null) {
            return vals;
        }
        List<T> left_vals = inorderTraversal(root.getLeft());
        if (left_vals.size() != 0) { vals.addAll(left_vals); }

        vals.add(root.getValue());

        List<T> right_vals = inorderTraversal(root.getRight());
        if (right_vals.size() != 0) { vals.addAll(right_vals); }

        return vals;
    }

    /**
     * 后序遍历 BinaryNode
     * @param root 根节点
     * @return 后序遍历的有序节点集合
     */
    public static <T> List<T> postorderTraversal(BinaryNode<T> root) {
        List<T> vals = new ArrayList<>();
        if (root == null) {
            return vals;
        }
        List<T> left_vals = postorderTraversal(root.getLeft());
        if (left_vals.size() != 0) { vals.addAll(left_vals); }

        List<T> right_vals = postorderTraversal(root.getRight());
        if (right_vals.size() != 0) { vals.addAll(right_vals); }

        vals.add(root.getValue());

        return vals;
    }

    /**
     * 广度优先搜索 遍历 BinaryNode
     * @param root 根节点
     * @return 按层返回每一节点的值
     */
    public static <T> List<T> bfsOrder(BinaryNode<T> root) {
        List<T> lists = new ArrayList<>();
        if (root == null) {
            return lists;
        }

        Queue<BinaryNode<T>> queue = new LinkedList<>();
        queue.offer(root);

        while (queue.size() != 0) {
            BinaryNode<T> curNode = queue.poll();
            lists.add(curNode.getValue());

            if (curNode.getLeft() != null) {
                queue.offer(curNode.getLeft());
            }
            if (curNode.getRight() != null) {
                queue.offer(curNode.getRight());
            }
        }

        return lists;
    }

    /**
     * 层次遍历 BinaryNode
     * @param root 根节点
     * @return 按层返回每一节点的值
     */
    public static <T> List<List<T>> levelOrder(BinaryNode<T> root) {
        List<List<T>> lists = new ArrayList<>();
        if (root == null) {
            return lists;
        }

        Queue<BinaryNode<T>> queue = new LinkedList<>();
        queue.offer(root);
        BinaryNode<T> nlast = root;
        BinaryNode<T> travelNode = null;

        List<T> curLevel = new ArrayList<>();
        while (queue.size() != 0) {
            BinaryNode<T> curNode = queue.poll();
            curLevel.add(curNode.getValue());

            if (curNode.getLeft() != null) {
                queue.offer(curNode.getLeft());
                travelNode = curNode.getLeft();
            }
            if (curNode.getRight() != null) {
                queue.offer(curNode.getRight());
                travelNode = curNode.getRight();
            }

            if (curNode == nlast) {
                lists.add(curLevel);

                nlast = travelNode;

                curLevel = new ArrayList<>();
            }
        }

        return lists;
    }


    /**
     * 自顶向下 计算 BinaryNode 的深度
     * @param root 根节点
     * @param depth 初始深度，根节点时为1
     * @param <T> 泛型类
     */
    public static <T> void maximumDepthByTopDown(BinaryNode<T> root, int depth) {
        if (root == null) {
            return ;
        }
        if (root.getLeft() == null && root.getRight() == null) {
            root.setDepth(Math.max(root.getDepth(), depth));
        }

        maximumDepthByTopDown(root.getLeft(), depth + 1);
        maximumDepthByTopDown(root.getRight(), depth + 1);
    }

    /**
     * 自底向上 计算 BinaryNode 的深度
     * @param root 根节点
     * @return 根节点的深度
     */
    public static <T> int maximumDepthByBottomUp(BinaryNode<T> root) {
        if (root == null) {
            return 0;
        }
        int left_depth = maximumDepthByBottomUp(root.getLeft());
        int right_depth = maximumDepthByBottomUp(root.getRight());

        return Math.max(left_depth, right_depth) + 1;
    }

    /**
     * 判断单个节点是否对称
     * @param root 节点
     * @param <T> 泛型类
     * @return 布尔值
     */
    public static <T> boolean isSymmetric(BinaryNode<T> root) {
        // 如果节点为空，直接返回true
        if (root == null) {
            return true;
        }
        // 判断该节点的左孩子和右孩子是否对称
        return isSymmetric(root.getLeft(), root.getRight());
    }

    /**
     * 比较两个节点是否对称
     * @param firstNode 第一个节点
     * @param secondNode 第二个节点
     * @param <T> 泛型类
     * @return 布尔值
     */
    public static <T> boolean isSymmetric(BinaryNode<T> firstNode, BinaryNode<T> secondNode) {
        // 如果两个节点都为空，直接返回true
        if (firstNode == null && secondNode == null) {
            return true;
        }
        // 如果两个节点都不为空
        else if(firstNode != null && secondNode != null) {
            // 判断连个节点是否满足一下三个条件：
            // 1、如果两个节点的值相等
            // 2、第一个节点的左孩子与第二个节点的右孩子对称
            // 3、第一个节点的右孩子与第二个节点的左孩子对称
            return firstNode.getValue() == secondNode.getValue() &&
                    isSymmetric(firstNode.getLeft(), secondNode.getRight()) &&
                    isSymmetric(firstNode.getRight(), secondNode.getLeft());
        }
        // 如果两个节点中又一个为空
        return false;
    }

    /**
     * 根据中序遍历和后序遍历构造 BinaryNode
     * @param inorder 中序遍历数组
     * @param postorder 后序遍历数组
     * @param <T> 泛型类
     * @return BinaryNode对象
     */
    public static <T> BinaryNode<T> buildTreeByInAndPost(T[] inorder, T[] postorder) {
        if (inorder.length == 0 || postorder.length == 0) {
            return null;
        }

        int inBegin = 0;
        int inEnd = inorder.length - 1;
        int postBegin = 0;
        int postEnd = postorder.length - 1;

        return treeHandlerByInAndPost(inorder, postorder, inBegin, inEnd, postBegin, postEnd);
    }

    /**
     * 根据中序遍历和后序遍历构造 BinaryNode 的工具方法
     * @param inorder 中序遍历数组
     * @param postorder 后序遍历数组
     * @param inBegin 中序遍历数组开始索引
     * @param inEnd 中序遍历数组结束索引
     * @param postBegin 后序遍历数组开始索引
     * @param postEnd 后序遍历数组结束索引
     * @param <T> 泛型类
     * @return BinaryNode对象
     */
    public static <T> BinaryNode<T> treeHandlerByInAndPost(T[] inorder, T[] postorder, int inBegin, int inEnd, int postBegin, int postEnd) {
        BinaryNode<T> root = new BinaryNode<T>(postorder[postEnd]);
        int pos = findIndex(inorder, postorder[postEnd]);

        int leftLength = pos - inBegin;
        int rightLength = inEnd - pos;

        if (leftLength > 0) {
            BinaryNode<T> leftNode = treeHandlerByInAndPost(inorder, postorder, inBegin, pos - 1, postBegin, postBegin + leftLength - 1);
            root.setLeft(leftNode);
        }
        if (rightLength > 0) {
            BinaryNode<T> rightNode = treeHandlerByInAndPost(inorder, postorder, pos + 1, inEnd, postBegin + leftLength, postEnd - 1);
            root.setRight(rightNode);
        }

        return root;
    }

    /**
     * 在数组中找到指定元素的下标，没有则返回-1
     * @param arr 数组
     * @param val 指定元素
     * @param <T> 泛型类
     * @return 索引
     */
    public static <T> int findIndex(T[] arr, T val) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == val) return i;
        }
        return -1;
    }

    /**
     * "自顶向下"获取两个节点在指定节点中的最近公共祖先
     * @param root 指定节点
     * @param p 第一个节点
     * @param q 第二个节点
     * @param <T> 泛型类
     * @return 最近公共祖先
     */
    public static <T> BinaryNode<T> lowestCommonAncestorByTopDown(BinaryNode<T> root, BinaryNode<T> p, BinaryNode<T> q) {
        if (search(root.getLeft(), p) && search(root.getLeft(), q)) return lowestCommonAncestorByTopDown(root.getLeft(), p, q);

        if (search(root.getRight(), p) && search(root.getRight(), q)) return lowestCommonAncestorByTopDown(root.getRight(), p, q);

        if (search(root, p) || search(root, q)) return null;

        return root;
    }

    /**
     * 在指定节点中搜索某一个节点
     * @param root 指定节点
     * @param p 带查找的节点
     * @param <T> 泛型类
     * @return 是否存在
     */
    public static <T> Boolean search(BinaryNode<T> root, BinaryNode<T> p) {
        if (root == null) return false;

        if (root == p) return true;

        return search(root.getLeft(), p) || search(root.getRight(), p);
    }

    /**
     * "自底向上"获取两个节点在指定节点中的最近公共祖先
     * @param root 指定节点
     * @param p 第一个节点
     * @param q 第二个节点
     * @param <T> 泛型类
     * @return 最近公共祖先
     */
    public static <T> BinaryNode<T> lowestCommonAncestorByBottomUp(BinaryNode<T> root, BinaryNode<T> p, BinaryNode<T> q) {
        if (root == null) return null;
        if (root == p || root == q) return root;

        BinaryNode<T> leftAncestor = lowestCommonAncestorByBottomUp(root.getLeft(), p, q);
        BinaryNode<T> rightAncestor = lowestCommonAncestorByBottomUp(root.getRight(), p, q);

        if (leftAncestor != null && rightAncestor != null) return root; // p和q分别在root的左右子节点

        return leftAncestor != null ? leftAncestor : rightAncestor;     // p和q在root左右子节点中的一侧，或不存在
    }

    /**
     * 使用"层次遍历"序列化 BinaryNode 对象
     * @param root BinaryNode 对象
     * @param <T> 泛型类
     * @return 序列化后的字符串
     */
    public static <T> String serialize(BinaryNode<T> root) {
        Queue<BinaryNode<T>> queue = new LinkedList<>();
        queue.offer(root);

        StringBuilder serialStr = new StringBuilder();
        serialStr.append("[");
        while (!queue.isEmpty()) {
            BinaryNode<T> curNode = queue.poll();
            if (curNode != null) {
                serialStr.append(curNode.getValue());
                queue.offer(curNode.getLeft());
                queue.offer(curNode.getRight());
            } else {
                serialStr.append("null");
            }
            serialStr.append(",");
        }
        serialStr.deleteCharAt(serialStr.length() - 1);
        serialStr.append("]");
        return serialStr.toString();
    }

    /**
     * 使用"层次遍历"反序列化为 BinaryNode 对象
     * @param data 序列化的字符串
     * @param <T> 泛型类
     * @return BinaryNode 对象
     */
    public static <T> BinaryNode<T> deserialize(String data) {
        if (null == data || data.isEmpty()) return null;

        String str = data.substring(1, data.length() - 1);
        String[] strs = str.split(",");

        int index = 0;
        BinaryNode<T> root = buildTree((T)strs[index++]);
        if (root == null) return root;

        Queue<BinaryNode<T>> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            BinaryNode<T> curNode = queue.poll();
            curNode.setLeft(buildTree((T)strs[index++]));
            curNode.setRight(buildTree((T)strs[index++]));
            if (curNode.getLeft() != null) queue.offer(curNode.getLeft());
            if (curNode.getRight() != null) queue.offer(curNode.getRight());
        }
        return root;
    }

    public static <T> BinaryNode<T> buildTree(T val) {
        if ("null".equals(val)) return null;

        return new BinaryNode<>(val);
    }
}