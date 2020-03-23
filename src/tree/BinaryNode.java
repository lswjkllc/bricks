package tree;

import javafx.beans.binding.StringBinding;

import java.util.Arrays;
import java.util.List;

public class BinaryNode<T> {
    private T value;
    private BinaryNode<T> left;
    private BinaryNode<T> right;
    private int depth = 0;

    public BinaryNode (T value) {
        this.value = value;
    }

    public T getValue() {
        return value;
    }

    public void setValue(T value) {
        this.value = value;
    }

    public BinaryNode<T> getLeft() {
        return left;
    }

    public void setLeft(BinaryNode<T> left) {
        this.left = left;
    }

    public BinaryNode<T> getRight() {
        return right;
    }

    public void setRight(BinaryNode<T> right) {
        this.right = right;
    }

    public int getDepth() {
        return depth;
    }

    public void setDepth(int depth) {
        this.depth = depth;
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("[");
        List<T> values = NodeUtil.bfsOrder(this);
        for (T val: values) {
            str.append(val.toString());
            str.append(",");
        }
        str.deleteCharAt(str.length() - 1);
        str.append("]");
        return str.toString();
    }
}