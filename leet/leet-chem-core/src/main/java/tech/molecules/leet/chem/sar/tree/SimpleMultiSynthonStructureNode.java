package tech.molecules.leet.chem.sar.tree;

import tech.molecules.leet.chem.sar.SimpleMultiSynthonStructure;
import tech.molecules.leet.chem.sar.SimpleSynthonSet;
import tech.molecules.leet.commons.tree.AbstractMappedTreeNode;

import javax.swing.tree.TreeNode;
import java.util.ArrayList;
import java.util.List;

public class SimpleMultiSynthonStructureNode extends AbstractMappedTreeNode<SimpleMultiSynthonStructure> {

    public SimpleMultiSynthonStructureNode(SimpleMultiSynthonStructure structure, TreeNode parent) {
        super(structure, parent);
    }

    @Override
    protected List<Object> getChildObjects() {
        return new ArrayList<>(nodeObject.getSynthonSets());
    }

    @Override
    protected TreeNode createNodeRepresentation(Object child) {
        if (child instanceof SimpleSynthonSet) {
            return new SimpleSynthonSetNode((SimpleSynthonSet) child, this);
        }
        return null;
    }
}

