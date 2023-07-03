package tech.molecules.leet.chem.sar.tree;

import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.combinatorialspace.Synthon;
import tech.molecules.leet.commons.tree.AbstractMappedTreeNode;

import javax.swing.tree.TreeNode;
import java.util.Collections;
import java.util.List;

public class SynthonNode extends AbstractMappedTreeNode<StereoMolecule> {

    public SynthonNode(StereoMolecule synthon, TreeNode parent) {
        super(synthon, parent);
    }

    @Override
    protected List<Object> getChildObjects() {
        return Collections.emptyList(); // Leaf node, no children
    }

    @Override
    protected TreeNode createNodeRepresentation(Object child) {
        return null; // Leaf node, no children
    }

    @Override
    public String toString() {
        return "Synthon "+getNodeObject().getIDCode();
    }
}

