package tech.molecules.leet.chem.sar.tree;

import tech.molecules.leet.chem.sar.SimpleSARProject;
import tech.molecules.leet.chem.sar.SimpleSARSeries;
import tech.molecules.leet.commons.tree.AbstractMappedTreeNode;

import javax.swing.tree.TreeNode;
import java.util.ArrayList;
import java.util.List;

public class SARProjectTreeNode extends AbstractMappedTreeNode<SimpleSARProject> {

    public SARProjectTreeNode(SimpleSARProject model, TreeNode parent) {
        super(model, parent);
    }

    @Override
    protected List<Object> getChildObjects() {
        return new ArrayList<>(nodeObject.getSeries());
    }

    @Override
    protected TreeNode createNodeRepresentation(Object child) {
        if (child instanceof SimpleSARSeries) {
            return new SARSeriesTreeNode((SimpleSARSeries) child, this);
        }
        return null;
    }

}
