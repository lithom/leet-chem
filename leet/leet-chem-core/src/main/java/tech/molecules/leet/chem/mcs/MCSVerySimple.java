package tech.molecules.leet.chem.mcs;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.SSSearcher;
import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.ChemUtils;

import java.util.*;

public class MCSVerySimple {


    public final static class BranchInfo {
        public final int[] m;

        //public final int a_candidate;
        public final BitSet candidates_A;
        public final BitSet forbidden_A;

        public final BitSet ta;
        public final BitSet tb;

        public final double score;

        public final int last_a;
        //public final
        public BranchInfo(int[] m, BitSet candidates_A, BitSet forbidden_A, BitSet ta, BitSet tb, double score, int last_a) {
            this.m = m;
            this.candidates_A = candidates_A;
            this.forbidden_A = forbidden_A;
            //this.a_candidate = a_candidate;
            this.ta = ta;
            this.tb = tb;
            this.score = score;
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
        this.candidateMap = new int[a.getAtoms()];
        Arrays.fill(candidateMap,-1);
        int[] m_start = new int[a.getAtoms()];
        Arrays.fill(m_start,-1);
        BitSet ta = new BitSet(a.getAtoms());
        BitSet tb = new BitSet(b.getAtoms());
        BitSet candidates_A = new BitSet(a.getAtoms());
        BitSet forbidden_A = new BitSet(a.getAtoms());
        //for(int zi=0;zi<A.getAtoms();zi++) {
        matchRecursiveNextComponent(new BranchInfo(m_start,candidates_A,forbidden_A,ta, tb, 0,-1));
        //matchRecursive(new BranchInfo(m_start,candidates_A,forbidden_A,ta, tb, 0,-1));
        //}
    }

    public void matchRecursiveNextComponent(BranchInfo b) {
        BitSet current_forbidden_A = b.forbidden_A;
        for(int zi=0;zi<A.getAtoms();zi++) {
            if(!current_forbidden_A.get(zi)) {
                BitSet candidates_A = new BitSet(A.getAtoms());
                candidates_A.set(zi);
                BranchInfo b_next = new BranchInfo(b.m,candidates_A,current_forbidden_A,b.ta,b.tb,b.score,b.last_a);
                matchIncrementalRecursive(b_next);
                current_forbidden_A.set(zi);
            }
        }
    }

    public void matchIncrementalRecursive(BranchInfo b) {

        if (b.candidates_A.isEmpty()) {
            updateCandidate(b);
            return;
        }

        // evaluate upper bound..
        if(upperBoundSimple(b) < candidateScore) {
            return;
        }

        int v1 = selectNextA(b);
        BitSet candidates_A_next = (BitSet) b.candidates_A.clone();
        candidates_A_next.set(v1,false);

        List<Integer> compatibleNodes = candidateMatchAtoms(b, v1,false);

        if(false) {
            System.out.println("Status: " + b.toString());
            System.out.println("Next v1:  " + v1);
            System.out.println("Candidates: " + compatibleNodes.toString());
        }

        for (int zi = 0; zi < compatibleNodes.size(); zi++) {
            int vb = compatibleNodes.get(zi);
            // ok, next map is v1 -> vb
            BranchInfo b_next = createNextBranchInfo(b, v1, vb);
            BitSet forbidden_A_next = updateForbiddenNodes(b_next,v1,vb);

            // now find the next candidate vertices in A:
            BitSet nc = nextCandidatesA(v1,candidates_A_next,forbidden_A_next);

            matchIncrementalRecursive(new BranchInfo(b_next.m,nc,forbidden_A_next,b_next.ta,b_next.tb,b_next.score,v1));
            //matchRecursive(b_next);
        }
        return;
    }
    public void matchRecursive(BranchInfo b) {
        if(upperBound(b) < candidateScore) {
            return;
        }

        List<Integer> v1_list = new ArrayList<>();
        if(b.last_a<0) {
            v1_list = order_root(b);
        }
        else {
            v1_list = order_branch(b);
        }

        if (v1_list.isEmpty()) {
            updateCandidate(b);
            return;
        }
        for(int z1=0;z1<v1_list.size();z1++){
            int v1 = v1_list.get(z1);
            //BitSet tnew = (BitSet) b.ta.clone();
            //tnew.set(v1);

            // loop over possible vertices in b:
            List<Integer> compatibleNodes = candidateMatchAtoms(b, v1, false);
            if (compatibleNodes.isEmpty()) {
                updateCandidate(b);
                return;
            }
            for (int zi = 0; zi < compatibleNodes.size(); zi++) {
                int vb = compatibleNodes.get(zi);
                BranchInfo b_next = createNextBranchInfo(b, v1, vb);
                matchRecursive(b_next);
            }
        }
    }

    private void updateCandidate(BranchInfo b) {
        // update candidate..
        if(b.score>candidateScore) {
            candidateMap = b.m;
            candidateScore = b.score;
        }
    }

    public static int selectNextA(BranchInfo b) {
        int next = b.candidates_A.nextSetBit(0);
        return next;
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
        return new BranchInfo(m_next,old.candidates_A,old.forbidden_A,ta_next,tb_next,old.score+1,v1);
    }

    public double upperBound(BranchInfo b) {
        return 100;
    }

    public double upperBoundSimple(BranchInfo b) {
        return b.score + (A.getAtoms()-b.forbidden_A.cardinality());
    }


    public List<Integer> order_root(BranchInfo b) {
        //int ret = -1;
        List<Integer> ret = new ArrayList<>();
        for(int zi=0;zi<A.getAtoms();zi++) {
            //if(!b.ta.get(zi)) {
            //    return zi;
            //}
            if(!b.ta.get(zi)) { ret.add(zi); }
        }
        return ret;
    }

    // returns only "larger number" neighbors of already matched atoms in A
    public List<Integer> order_branch(BranchInfo b) {
        //int ret = -1;
        Set<Integer> ret = new HashSet<>();
        for(int za : b.ta.stream().toArray()) {
            for (int zi = 0; zi < A.getConnAtoms(za); zi++) {
                int ca = A.getConnAtom(za, zi);
                if (ca > b.last_a) {
                    ret.add(ca);
                }
                //if(!b.ta.get(zi)) {
                //    return zi;
                //}
                //if(!b.ta.get(zi)) { ret.add(zi); }
            }
        }
        return new ArrayList<>(ret);
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
                for( int zj=0;zj<A.getConnAtoms(v1);zj++) {
                    int a_neighbor = A.getConnAtom(v1,zj);
                    int mapped_in_B = b.m[a_neighbor];
                    if(mapped_in_B<0) {
                        continue;
                    }
                    // search for it in neighbors of B
                    int b_bond = B.getBond(zi,mapped_in_B);
                    if(b_bond<0) {
                        compat = false;
                        break;
                    }

                    // check that bonds are compatible!..
                    if(!compareBonds( A.getBond(v1,a_neighbor) ,b_bond)) {
                        compat = false;
                        break;
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
        // TODO: maybe now sort according to "similarity"? Probably..
        return la;
    }

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

    public static void main(String args[]) {
        //StereoMolecule a_1 = ChemUtils.parseSmiles("FC(C1CC1)N(CC1)CC1F");
        StereoMolecule a_1 = ChemUtils.parseSmiles("N#Cc1c(NCc2ccccc2)snc1N1CCCC1");
        //StereoMolecule b_1 = ChemUtils.parseSmiles("FC1C(CN(CCC2)C2F)C1");
        StereoMolecule b_1 = ChemUtils.parseSmiles("N#Cc1c(NC(c2cnccc2)F)snc1C1CCCC1");

        MCSVerySimple mcs = new MCSVerySimple();
        long ts_a = System.currentTimeMillis();
        mcs.match(a_1,b_1);
        long ts_b = System.currentTimeMillis();
        System.out.println("Time: "+(ts_b-ts_a));

        // check correctness..
        StereoMolecule mcs_a = new StereoMolecule();
        StereoMolecule mcs_b = new StereoMolecule();
        boolean mcs1[] = new boolean[a_1.getAtoms()];
        boolean mcs2[] = new boolean[b_1.getAtoms()];
        for(int zi=0;zi<mcs.candidateMap.length;zi++) {
            if(mcs.candidateMap[zi]>=0) {
                mcs1[zi] = true;
                mcs2[mcs.candidateMap[zi]] = true;
            }
        }
        a_1.copyMoleculeByAtoms(mcs_a,mcs1,true,null);
        b_1.copyMoleculeByAtoms(mcs_b,mcs2,true,null);

        System.out.println("a: "+mcs_a.getIDCode());
        System.out.println("b: "+mcs_b.getIDCode());

        System.out.println("mkay..");
    }


}
