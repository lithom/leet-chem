package tech.molecules.leet.commons.tree;

import javax.swing.tree.TreeNode;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.stream.Collectors;

public abstract class AbstractMappedTreeNode<T> implements TreeNode {

    protected T nodeObject;
    protected TreeNode parent;

    public AbstractMappedTreeNode(T nodeObject, TreeNode parent) {
        this.nodeObject = nodeObject;
        this.parent = parent;
    }

    public T getNodeObject() {return nodeObject;}

    @Override
    public TreeNode getChildAt(int childIndex) {
        List<Object> childObjects = getChildObjects();
        if (childIndex >= 0 && childIndex < childObjects.size()) {
            return createNodeRepresentation(childObjects.get(childIndex));
        }
        return null;
    }

    @Override
    public int getChildCount() {
        return getChildObjects().size();
    }

    @Override
    public TreeNode getParent() {
        return parent;
    }

    @Override
    public boolean getAllowsChildren() {
        return true;
    }

    @Override
    public boolean isLeaf() {
        return getChildObjects().isEmpty();
    }

    @Override
    public Enumeration<? extends TreeNode> children() {
        return Collections.enumeration(getChildObjects().stream()
                .map(this::createNodeRepresentation)
                .collect(Collectors.toList()));
    }

    @Override
    public int getIndex(TreeNode node) {
        List<Object> childObjects = getChildObjects();
        for (int i = 0; i < childObjects.size(); i++) {
            TreeNode childNode = createNodeRepresentation(childObjects.get(i));
            if (childNode.equals(node)) {
                return i;
            }
        }
        return -1;
    }

    @Override
    public String toString() {
        return this.nodeObject.toString();
    }

    protected abstract List<Object> getChildObjects();
    protected abstract TreeNode createNodeRepresentation(Object child);
}
