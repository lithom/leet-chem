package tech.molecules.leet.chem.mcs;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.ChemUtils;

import javax.swing.*;
import java.util.*;
import java.util.stream.Collectors;

public class MCS3 {

    private MCSStrategy strategy;

    public MCS3(MCSStrategy strategy) {
        this.strategy = strategy;
    }

    public final static class BranchInfo {
        public final int[] m;

        //public final int a_candidate;
        public final BitSet candidates_A;
        public final BitSet forbidden_A;

        public final BitSet ta;
        public final BitSet tb;

        public final int[] components_A;

        public final int[] components_B;

        public final double score;

        public final int[] componentSizes;

        /**
         * E.g. if numComponents is = 2, then we already have labeled vertices with 0 and
         * we are currently labeling vertices with 1.
         */
        public final int numComponents;
        public final int last_a;
        //public final
        public BranchInfo(int[] m, BitSet candidates_A, BitSet forbidden_A, BitSet ta, BitSet tb, int[] components_A, int[] components_B, int[] componentSizes, double score, int num_components, int last_a) {
            this.m = m;
            this.candidates_A = candidates_A;
            this.forbidden_A = forbidden_A;
            //this.a_candidate = a_candidate;
            this.ta = ta;
            this.tb = tb;
            this.components_A = components_A;
            this.components_B = components_B;
            this.componentSizes = componentSizes;
            this.score = score;
            this.numComponents = num_components;
            this.last_a = last_a;
        }
        public String toString() {
            List<String> matched_atoms = new ArrayList<>();
            for(int zi=0;zi<m.length;zi++) {
                if(m[zi]>=0) {
                    matched_atoms.add(""+zi+"->"+m[zi]);
                }
            }
            return "Score="+score+"\n"+String.join(",",matched_atoms);
        }
    }


    private StereoMolecule A;
    private StereoMolecule B;


    private double candidateScore;
    private int[]  candidateMap;

    public void match(StereoMolecule a, StereoMolecule b) {
        a.ensureHelperArrays(Molecule.cHelperCIP);
        b.ensureHelperArrays(Molecule.cHelperCIP);
        this.A = a;
        this.B = b;

        this.strategy.init(A,B);

        this.candidateMap = new int[a.getAtoms()];
        Arrays.fill(candidateMap,-1);
        int[] m_start = new int[a.getAtoms()];
        Arrays.fill(m_start,-1);
        BitSet ta = new BitSet(a.getAtoms());
        BitSet tb = new BitSet(b.getAtoms());
        BitSet candidates_A = new BitSet(a.getAtoms());
        BitSet forbidden_A = new BitSet(a.getAtoms());

        int[] components_A = new int[a.getAtoms()];
        int[] components_B = new int[b.getAtoms()];
        Arrays.fill(components_A,-1);
        Arrays.fill(components_B,-1);
        int[] componentSizes = new int[]{0,0,0,0,0,0,0,0};
        //for(int zi=0;zi<A.getAtoms();zi++) {
        matchRecursiveNextComponent(new BranchInfo(m_start,candidates_A,forbidden_A,ta, tb,components_A,components_B ,componentSizes,0,0,-1));
        //matchRecursive(new BranchInfo(m_start,candidates_A,forbidden_A,ta, tb, 0,-1));
        //}
    }

    public void matchRecursiveNextComponent(BranchInfo b) {
        BitSet current_forbidden_A = b.forbidden_A;
        int num_components = b.numComponents+1;

        if(true) {
            while(current_forbidden_A.cardinality() < A.getAtoms()) {
                BitSet candidates = new BitSet(A.getAtoms());
                candidates.or(current_forbidden_A);
                candidates.flip(0,A.getAtoms());
                int nextA = strategy.selectNextA(candidates);

                BitSet candidates_A = new BitSet(A.getAtoms());
                candidates_A.set(nextA);
                BranchInfo b_next = new BranchInfo(b.m, candidates_A, current_forbidden_A, b.ta, b.tb, b.components_A, b.components_B, b.componentSizes, b.score, num_components, b.last_a);
                matchIncrementalRecursive(b_next);
                current_forbidden_A.set(nextA);
            }
        }


        if(false) {
            for (int zi = 0; zi < A.getAtoms(); zi++) {
                if (!current_forbidden_A.get(zi)) {
                    BitSet candidates_A = new BitSet(A.getAtoms());
                    candidates_A.set(zi);
                    BranchInfo b_next = new BranchInfo(b.m, candidates_A, current_forbidden_A, b.ta, b.tb, b.components_A, b.components_B, b.componentSizes, b.score, num_components, b.last_a);
                    matchIncrementalRecursive(b_next);
                    current_forbidden_A.set(zi);
                }
            }
        }
    }

    public void matchIncrementalRecursive(BranchInfo b) {

        if (b.candidates_A.isEmpty()) {
            if(strategy.checkComponentConstraintsValid(b)) {
                if(strategy.checkStartNextComponent(b)) {
                    matchRecursiveNextComponent(b);
                }
            }

            updateCandidate(b);
            return;
        }

        // evaluate upper bound..
        if(strategy.upperBound(b) < candidateScore) {
            return;
        }

        int v1 = strategy.selectNextA(b);
        BitSet candidates_A_next = (BitSet) b.candidates_A.clone();
        candidates_A_next.set(v1,false);

        List<Integer> compatibleNodesPre = candidateMatchAtoms(b, v1,false);
        int[] compatibleNodesSorted = strategy.sortCandidatesB(v1,compatibleNodesPre);


        if(false) {
            System.out.println("Status: " + b.toString());
            System.out.println("Status_curerent_best:  " + candidateScore);
            System.out.println("Next v1:  " + v1);
            System.out.println("Candidates: " + compatibleNodesSorted.toString());
        }

        for (int zi = 0; zi < compatibleNodesSorted.length; zi++) {
            int vb = compatibleNodesSorted[zi];
            // ok, next map is v1 -> vb
            BranchInfo b_next = createNextBranchInfo(b, v1, vb);
            BitSet forbidden_A_next = updateForbiddenNodes(b_next,v1,vb);

            // now find the next candidate vertices in A:
            BitSet nc = strategy.nextCandidatesA(v1,candidates_A_next,forbidden_A_next);

            matchIncrementalRecursive(new BranchInfo(b_next.m,nc,forbidden_A_next,b_next.ta,b_next.tb,b_next.components_A,b_next.components_B,b_next.componentSizes,b_next.score,b_next.numComponents,v1));
            //matchRecursive(b_next);
        }
        return;
    }

    private void updateCandidate(BranchInfo b) {
        // update candidate..
        if(b.score>candidateScore) {
            candidateMap = b.m;
            candidateScore = b.score;
        }
    }

    public BitSet updateForbiddenNodes(BranchInfo b, int va, int vb) {
        BitSet forbidden_next = (BitSet) b.forbidden_A.clone();
        forbidden_next.set(va);

        // loop over unforbidden vertices and check if there is possible mapping partner:
        for(int zi=0;zi<A.getAtoms();zi++) {
            if(!forbidden_next.get(zi)){
                if( candidateMatchAtoms(b,zi,true).isEmpty() ) {
                    forbidden_next.set(zi);
                }
            }
        }
        return forbidden_next;
    }

    public static BranchInfo createNextBranchInfo(BranchInfo old, int v1, int vb) {
        int[] m_next = Arrays.copyOf(old.m,old.m.length);
        m_next[v1] = vb;
        BitSet ta_next = (BitSet) old.ta.clone();
        BitSet tb_next = (BitSet) old.tb.clone();
        ta_next.set(v1);
        tb_next.set(vb);
        int[] components_A = Arrays.copyOf(old.components_A,old.components_A.length);
        int[] components_B = Arrays.copyOf(old.components_A,old.components_B.length);
        components_A[v1] = old.numComponents-1;
        components_B[vb] = old.numComponents-1;
        int[] component_sizes = Arrays.copyOf(old.componentSizes,old.componentSizes.length);
        component_sizes[old.numComponents-1]++;
        return new BranchInfo(m_next,old.candidates_A,old.forbidden_A,ta_next,tb_next,components_A,components_B,component_sizes,old.score+1,old.numComponents,v1);
    }

    public List<Integer> candidateMatchAtoms(BranchInfo b, int v1, boolean stopAfterFirst) {
        List<Integer> la = new ArrayList<>();
        for(int zi=0;zi<B.getAtoms();zi++) {
            if(!b.tb.get(zi)) {

                // check that atom is the same
                if(A.getAtomicNo(v1)!=B.getAtomicNo(zi)) {
                    //compat = false;
                    continue;
                }

                // check compatiblity, i.e. check if all matched neighbors of v1 in A are matched to neighbors of zi in B
                boolean compat = true;
                if(true) {
                    for (int zj = 0; zj < A.getConnAtoms(v1); zj++) {
                        int a_neighbor = A.getConnAtom(v1, zj);
                        int mapped_in_B = b.m[a_neighbor];
                        if (mapped_in_B < 0) {
                            continue;
                        }
                        // search for it in neighbors of B
                        // NOTE: we checked if first just testing for neighbors and then getting the bond is faster. It
                        // was not significantly faster.
                        int b_bond = B.getBond(zi, mapped_in_B);
                        if (b_bond < 0) {
                            compat = false;
                            break;
                        }

                        // check that bonds are compatible!..
                        if (!strategy.compareBonds(A.getBond(v1, a_neighbor), b_bond)) {
                            compat = false;
                            break;
                        }
                    }
                    if(compat) {
                        // check for all neighbors of zi in B if we find the correctly mapped neighbors
                        //Map<Integer,Integer> m_inverse = ChemUtils.inverseMap(b.m);
                        //m_inverse.put(zi,v1);
                        int[] m_inverse = ChemUtils.inverseMap2(b.m,A.getAtoms(),B.getAtoms());
                        for(int zj = 0; zj < B.getConnAtoms(zi) ; zj++) {
                            int b_neighbor = B.getConnAtom(zi,zj);
                            int mapped_in_A = m_inverse[b_neighbor];
                            if(zj==zi) {mapped_in_A=v1;}
                            //Integer mapped_in_A = m_inverse.get(b_neighbor);
                            //if (mapped_in_A == null) {
                            if(mapped_in_A <0) {
                                continue;
                            }
                            // search for it in neighbors of A
                            int a_bond = A.getBond(v1, mapped_in_A);
                            if (a_bond < 0) {
                                compat = false;
                                break;
                            }
                            // check that bonds are compatible!..
                            if (!strategy.compareBonds(a_bond, B.getBond(zi, b_neighbor))) {
                                compat = false;
                                break;
                            }
                        }
                    }
                }
                if(false) {
                    for (int zj = 0; zj < A.getConnAtoms(v1); zj++) {
                        int a_neighbor = A.getConnAtom(v1, zj);
                        int mapped_in_B = b.m[a_neighbor];
                        if (mapped_in_B < 0) {
                            continue;
                        }
                        // search for it in neighbors of B
                        int b_bond = B.getBond(zi, mapped_in_B);
                        if (b_bond < 0) {
                            compat = false;
                            break;
                        }

                        // check that bonds are compatible!..
                        if (!strategy.compareBonds(A.getBond(v1, a_neighbor), b_bond)) {
                            compat = false;
                            break;
                        }
                    }
                }
                if(compat) {
                    la.add(zi);
                    if(stopAfterFirst) {
                        break;
                    }
                }
            }
        }
        return la;
    }

    public static void computeMCS(StereoMolecule a_1, StereoMolecule b_1, MCS3 mcs) {
        //MCS3 mcs = new MCS3(new MCSDefaultStrategy());
        //MCS3 mcs_disconnected = new MCS3(new MCSDisconnectedDefaultStrategy());

        long ts_a = System.currentTimeMillis();
        mcs.match(a_1,b_1);
        long ts_b = System.currentTimeMillis();
        System.out.println("Time: "+(ts_b-ts_a));

//        long ts2_a = System.currentTimeMillis();
//        mcs_disconnected.match(a_1,b_1);
//        long ts2_b = System.currentTimeMillis();
//        System.out.println("Time (disconnected): "+(ts2_b-ts2_a));

        // check correctness..
        System.out.println("solution:");
        StereoMolecule mcsx_a = new StereoMolecule();
        StereoMolecule mcsx_b = new StereoMolecule();
        boolean mcsx1[] = new boolean[a_1.getAtoms()];
        boolean mcsx2[] = new boolean[b_1.getAtoms()];
        for(int zi=0;zi<mcs.candidateMap.length;zi++) {
            if(mcs.candidateMap[zi]>=0) {
                mcsx1[zi] = true;
                mcsx2[mcs.candidateMap[zi]] = true;
            }
        }
        a_1.copyMoleculeByAtoms(mcsx_a,mcsx1,true,null);
        b_1.copyMoleculeByAtoms(mcsx_b,mcsx2,true,null);

        System.out.println("a: "+mcsx_a.getIDCode());
        System.out.println("b: "+mcsx_b.getIDCode());

//        System.out.println("disjoint solution:");
//        StereoMolecule mcs_a = new StereoMolecule();
//        StereoMolecule mcs_b = new StereoMolecule();
//        boolean mcs1[] = new boolean[a_1.getAtoms()];
//        boolean mcs2[] = new boolean[b_1.getAtoms()];
//        for(int zi=0;zi<mcs_disconnected.candidateMap.length;zi++) {
//            if(mcs_disconnected.candidateMap[zi]>=0) {
//                mcs1[zi] = true;
//                mcs2[mcs_disconnected.candidateMap[zi]] = true;
//            }
//        }
//        a_1.copyMoleculeByAtoms(mcs_a,mcs1,true,null);
//        b_1.copyMoleculeByAtoms(mcs_b,mcs2,true,null);
//
//        System.out.println("a: "+mcs_a.getIDCode());
//        System.out.println("b: "+mcs_b.getIDCode());
//
//        System.out.println("mkay..");
    }

    public static void main(String args[]) {

        runBenchmark_A();

        if(false) {

            StereoMolecule a_1 = ChemUtils.parseSmiles("FC(C1CC1)N(CC1)CC1F");
            StereoMolecule b_1 = ChemUtils.parseSmiles("FC1C(CN(CCC2)C2F)C1");

            StereoMolecule a_2 = ChemUtils.parseSmiles("N#Cc1c(NCc2ccccc2)snc1N1CCCC1");
            StereoMolecule b_2 = ChemUtils.parseSmiles("N#Cc1c(NC(c2cnccc2)F)snc1C1CCCC1");

            StereoMolecule a_3 = ChemUtils.parseSmiles("CCCc1nn(-c2cc(NCCn(c(Cl)c3)c4c3c(OC)ccc4F)ncn2)nc1C(N(C)C)=O");
            StereoMolecule b_3 = ChemUtils.parseSmiles("CN(C)C(CCn1ncc(-c2cc(NCCn(c(C#N)cc3c(cc4)OC)c3c4F)ncn2)c1)=O");

            StereoMolecule a_4 = ChemUtils.parseSmiles("CN(C)C(c1nn(-c2cc(NCCn(c(Cl)c3)c4c3c(OC)ccc4F)ncn2)nc1C(CC1)C=C1Nc(snc1)c1C#N)=O");
            StereoMolecule b_4 = ChemUtils.parseSmiles("CN(C)C(c(c(C1N=NC(Nc(snc2)c2C#N)=N1)n1)nn1OC(C1)C1NCCn(c(Cl)c1)c2c1c(OC)ccc2F)=O");

            //MCS3 mcs = new MCS3(new MCSDefaultStrategy());
            //computeMCS(a_4,b_4,mcs);

            MCS3 mcs_dc = new MCS3(new MCSDisconnectedDefaultStrategy());
            computeMCS(a_4, b_4, mcs_dc);
        }

    }

    public static void runBenchmark_A() {
        List<StereoMolecule> mols = MCS2.getSomeTestMolecules(4000).stream().filter(mi -> mi.getAtoms()<40 && mi.getAtoms()>3 ).collect(Collectors.toList());

        long ts_a = System.currentTimeMillis();
        for(int za = 0; za < 1; za++) {
            for (int zi = 0; zi < 20; zi++) {
                MCS3 mcs = new MCS3(new MCSDisconnectedDefaultStrategy());//new MCS3(new MCSDefaultStrategy());
                StereoMolecule ma = mols.get(2 * zi);
                StereoMolecule mb = mols.get(2 * zi + 1);
                ma.stripSmallFragments();
                mb.stripSmallFragments();
                System.out.println("mcs: " + ma.getIDCode() + " / " + mb.getIDCode());
                computeMCS(ma, mb, mcs);
            }
        }
        long ts_b = System.currentTimeMillis();
        System.out.println("Time: "+(ts_b-ts_a));
    }


}
