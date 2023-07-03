package tech.molecules.leet.chem.sar.tree;

import tech.molecules.leet.chem.sar.SimpleMultiSynthonStructure;
import tech.molecules.leet.chem.sar.SimpleSARSeries;
import tech.molecules.leet.commons.tree.AbstractMappedTreeNode;

import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreeNode;
import java.util.ArrayList;
import java.util.List;

public class SARSeriesTreeNode extends AbstractMappedTreeNode<SimpleSARSeries> {

    public SARSeriesTreeNode(SimpleSARSeries series, TreeNode parent) {
        super(series, parent);
    }

    @Override
    protected List<Object> getChildObjects() {
        List<Object> children = new ArrayList<>();
        children.add(nodeObject.getSeriesDecomposition());
        children.addAll(nodeObject.getLabels());
        return children;
    }

    @Override
    protected TreeNode createNodeRepresentation(Object child) {
        if (child instanceof SimpleMultiSynthonStructure) {
            //return new DefaultMutableTreeNode("MultiSynthonStructure");
            SimpleMultiSynthonStructure ci = (SimpleMultiSynthonStructure) child;
            if( ci.getSynthonSets().size()==1 && ci.getSynthonSets().get(0).getSynthons().size()==1) {
                return new SynthonNode(ci.getSynthonSets().get(0).getSynthons().get(0),this);
            }
            else {
                return new SimpleMultiSynthonStructureNode((SimpleMultiSynthonStructure) child, this);
            }
        } else if (child instanceof String) {
            return new SARPartTreeNode((String) child, this);
        }
        return null;
    }

}
