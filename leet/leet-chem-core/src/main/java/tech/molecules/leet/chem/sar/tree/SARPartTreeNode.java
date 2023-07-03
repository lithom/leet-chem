package tech.molecules.leet.chem.sar.tree;
import tech.molecules.leet.commons.tree.AbstractMappedTreeNode;

import javax.swing.tree.TreeNode;
import java.util.Collections;
import java.util.List;

public class SARPartTreeNode extends AbstractMappedTreeNode<String> {

    public SARPartTreeNode(String part, TreeNode parent) {
        super(part, parent);
    }

    @Override
    protected List<Object> getChildObjects() {
        return Collections.emptyList(); // Leaf node, no children
    }

    @Override
    protected TreeNode createNodeRepresentation(Object child) {
        return null; // Leaf node, no children
    }
}
