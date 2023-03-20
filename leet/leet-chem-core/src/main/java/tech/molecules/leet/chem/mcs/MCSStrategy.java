package tech.molecules.leet.chem.mcs;

import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.ChemUtils;

import java.util.BitSet;
import java.util.List;

public interface MCSStrategy {

    public void init(StereoMolecule a, StereoMolecule b);

    public double upperBound(MCS3.BranchInfo bi);

    public boolean checkComponentConstraintsValid(MCS3.BranchInfo bi);

    public boolean checkStartNextComponent(MCS3.BranchInfo bi);

    public BitSet nextCandidatesA(int v, BitSet current_candidates, BitSet f_next);

    /**
     * aka "selection strategy 1"
     * @param bi
     * @return
     */
    public int selectNextA(MCS3.BranchInfo bi);

    public int selectNextA(BitSet candidates);


    public int[] sortCandidatesB(int v, BitSet ci);

    default public int[] sortCandidatesB(int v, List<Integer> ci) {
        return sortCandidatesB(v,ChemUtils.toBitSet(ci));
    }

    public int selectionStrategy0(int vmax, BitSet f);


    public boolean compareBonds(int bond_a, int bond_b);

}
