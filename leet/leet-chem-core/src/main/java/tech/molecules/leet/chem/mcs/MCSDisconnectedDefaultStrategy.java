package tech.molecules.leet.chem.mcs;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.descriptor.FragmentAtomFPHandler;

import java.util.BitSet;

public class MCSDisconnectedDefaultStrategy implements MCSStrategy {

    private StereoMolecule A, B;

    private AtomDescriptorSimilarityHelper adsh = new AtomDescriptorSimilarityHelper();

    @Override
    public void init(StereoMolecule a, StereoMolecule b) {
        this.A = new StereoMolecule(a);
        this.B = new StereoMolecule(b);
        this.A.ensureHelperArrays(Molecule.cHelperCIP);
        this.B.ensureHelperArrays(Molecule.cHelperCIP);

        long ts_a = System.currentTimeMillis();
        FragmentAtomFPHandler afph = new FragmentAtomFPHandler(6,6,512);
        adsh.setAtomDescriptors(afph.createDescriptor(this.A),afph.createDescriptor(this.B));
        long ts_b = System.currentTimeMillis();
        System.out.println("descriptorcalc: "+(ts_b-ts_a));
    }

    @Override
    public double upperBound(MCS3.BranchInfo b) {
        return b.score + (A.getAtoms()-b.forbidden_A.cardinality());
    }

    @Override
    public boolean checkComponentConstraintsValid(MCS3.BranchInfo bi) {
        return bi.numComponents<3;
    }

    @Override
    public boolean checkStartNextComponent(MCS3.BranchInfo bi) {
        //only if all components have min size
        boolean all_have_minsize = true;
        int min_component_sizes = 2;
        for(int zi=0;zi<bi.componentSizes.length;zi++) {
            if(bi.componentSizes[zi]>0 && bi.componentSizes[zi]<min_component_sizes) {
                return false;
            }
        }
        return true;
    }

    @Override
    public int selectNextA(MCS3.BranchInfo b) {
        if(true) { // selection using atom descriptors
            if(b.candidates_A.cardinality()==1) {return b.candidates_A.nextSetBit(0);}
            return adsh.findBestCandidateInA(b.candidates_A);
        }
        if(false) { // no selection
            int next = b.candidates_A.nextSetBit(0);
            return next;
        }
        return -1;
    }

    @Override
    public int selectNextA(BitSet candidates) {
        if(candidates.cardinality()==1) {return candidates.nextSetBit(0);}
        return adsh.findBestCandidateInA(candidates);
    }

    @Override
    public int[] sortCandidatesB(int v, BitSet ci) {
        if(true) {
            if(ci.cardinality()==1) {return new int[]{ci.nextSetBit(0)};}
            return adsh.findBestCandidateInB(v,ci);
        }
        return ci.stream().toArray();
    }

    @Override
    public int selectionStrategy0(int vmax, BitSet f) {
        return 0;
    }


    public BitSet nextCandidatesA(int v,BitSet candidates_next, BitSet f_next) {
        //int ret = -1;
        BitSet ret = (BitSet) candidates_next.clone();//new BitSet(A.getAtoms());
        ret.andNot(f_next);
        //for(int za : b.ta.stream().toArray()) {
        for (int zi = 0; zi < A.getConnAtoms(v); zi++) {
            int ca = A.getConnAtom(v, zi);
            if(!f_next.get(ca)) {
                ret.set(ca);
            }
        }
        return ret;
    }

    @Override
    public boolean compareBonds(int a_bond, int b_bond) {
        int a_order = A.getBondOrder(a_bond);
        int b_order = B.getBondOrder(b_bond);
        if(a_order == b_order) {
            return true;
        }
        boolean a_deloc = A.isDelocalizedBond(a_bond);
        boolean b_deloc = B.isDelocalizedBond(b_bond);
        if(a_deloc&&b_deloc) {return true;}
        if(a_deloc && (b_order==1 || b_order==2) ) {return true;}
        if(b_deloc && (a_order==1 || a_order==2) ) {return true;}

        return false;
    }

}
