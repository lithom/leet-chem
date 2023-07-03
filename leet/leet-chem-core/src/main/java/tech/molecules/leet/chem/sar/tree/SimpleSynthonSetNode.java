package tech.molecules.leet.chem.sar.tree;

import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.combinatorialspace.Synthon;
import tech.molecules.leet.chem.sar.SimpleSynthonSet;
import tech.molecules.leet.commons.tree.AbstractMappedTreeNode;

import javax.swing.tree.TreeNode;
import java.util.ArrayList;
import java.util.List;

public class SimpleSynthonSetNode extends AbstractMappedTreeNode<SimpleSynthonSet> {

    public SimpleSynthonSetNode(SimpleSynthonSet synthonSet, TreeNode parent) {
        super(synthonSet, parent);
    }

    @Override
    protected List<Object> getChildObjects() {
        return new ArrayList<>(nodeObject.getSynthons());
    }

    @Override
    protected TreeNode createNodeRepresentation(Object child) {
        if (child instanceof StereoMolecule) {
            return new SynthonNode((StereoMolecule) child, this);
        }
        return null;
    }
}
