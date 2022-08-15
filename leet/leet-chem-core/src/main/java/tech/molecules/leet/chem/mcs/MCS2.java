package tech.molecules.leet.chem.mcs;

import com.actelion.research.chem.Canonizer;
import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.SSSearcher;
import com.actelion.research.chem.StereoMolecule;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;
import tech.molecules.leet.chem.ChemUtils;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public abstract class MCS2 {

    protected StereoMolecule a;
    protected StereoMolecule b;

    //protected StereoMolecule getA() {return a;}
    //protected StereoMolecule getB() {return b;}

    /**
     * number of vertices in a and b
     */
    protected int nva;
    protected int nvb;

    /**
     * number of bonds in a and b
     */
    protected int nba;
    protected int nbb;

    /**
     * compatible vertices
     * entry (i x nvb) + j describes vertices a(i) and b(j)
     */
    protected BitSet vok;


    public void setAB(StereoMolecule ma, StereoMolecule mb) {
        this.a = ma;
        this.b = mb;
        this.a.ensureHelperArrays(Molecule.cHelperCIP);
        this.b.ensureHelperArrays(Molecule.cHelperCIP);
        this.nva = this.a.getAtoms();
        this.nvb = this.b.getAtoms();
        this.nba = this.a.getBonds();
        this.nbb = this.b.getBonds();
        // init vok:
        this.vok = new BitSet();
        for(int zi=0;zi<this.a.getAtoms();zi++) {
            for(int zj=0;zj<this.b.getAtoms();zj++) {
                vok.set(zi*nvb+zj , this.a.getAtomicNo(zi)==this.b.getAtomicNo(zj) );
            }
        }
    }

    public void computeMCS() {
        int ma[] = new int[nva];
        Arrays.fill(ma,-1);
        int maxmatchingsize = proceedWithNextComponent(ma,new BitSet(),0,0);
        System.out.println("max matching size: "+maxmatchingsize);
    }

    /**
     *
     * @param m map a to b
     * @param f
     * @param curmax
     */
    private int proceedWithNextComponent(int m[], BitSet f, int curmax, int ncomp) {

        for(int zi=0;zi<nva;zi++) {
            if(f.get(zi)) {continue;}
            int xi = selectionStrategy0(m.length,f);
            BitSet va = new BitSet(m.length);
            va.set(xi);
            curmax = mcsIncrement(m,va,f,curmax,ncomp+1);
            f.set(xi);
        }
        return curmax;
    }

    private int mcsIncrement(int m[], BitSet c_, BitSet f, int curmax, int ncomp) {
        if(c_.isEmpty()) {
            if(checkComponentConstraintsValid(m)) {
                // for the moment we break..
                curmax = Math.max( curmax , (int) Arrays.stream(m).filter( mi -> mi>=0).count() );
            }
            if(curmax>=8) {
                //System.out.println("mkay..");
            }
            if(true) {
                printDebug2(m,curmax);
            }
            return curmax;
        }
        int vi = selectionStrategy1(c_);
        BitSet c = (BitSet) c_.clone();
        c.set(vi,false);
        BitSet compatible = selectCompatibleNodes(m,vi,f);

        // we need this then in the recursion below:
        int upperbound2 = computeUpperBound(m,c,f);

        if(compatible.cardinality()>0) {
            int[] sorted_p = selectionStrategy2(vi, compatible);

            //printDebug(m,compatible,f,vi,curmax,upperbound2);

            for (int zi = 0; zi < sorted_p.length; zi++) {
                int pi = sorted_p[zi];
                BitSet fnext = updateForbiddenNodes(m, f, vi, pi);
                BitSet cnext = (BitSet) c.clone();
                for (int zc = 0; zc < a.getConnAtoms(vi); zc++) {
                    if (!fnext.get(a.getConnAtom(vi, zc))) {
                        cnext.set(a.getConnAtom(vi, zc));
                    }
                }
                int mnext[] = new int[m.length];
                for (int zm = 0; zm < m.length; zm++) {
                    mnext[zm] = m[zm];
                }
                mnext[vi] = pi;

                int upperbound = computeUpperBound(mnext, cnext, fnext);

                if (curmax < upperbound) {
                    curmax = mcsIncrement(mnext, cnext, fnext, curmax, ncomp);

                    if (upperbound2 <= curmax) {
                        break; // TODO: check if this is not actually "continue"??
                    }
                }
            }
        }
        BitSet f2 = (BitSet) f.clone();
        f2.set(vi,true);
        return mcsIncrement(m,(BitSet) c.clone(),f2,curmax,ncomp);
    }


    protected int computeUpperBound(int[] m, BitSet c, BitSet f) {
        int ub = 0;
        for(int zi=0;zi<m.length;zi++) {
            ub +=  ( m[zi]>=0 || !f.get(zi) )?1:0;
        }
        return ub;
    }

    /**
     * molecule,symmetry ranks, atom map
     */
    private Map<BitSet, Triple<StereoMolecule,int[],int[]>> A_subgraphs = new HashMap<>();
    private Map<BitSet, Triple<StereoMolecule,int[],int[]>> B_subgraphs = new HashMap<>();


    private Map<Integer,List<Integer>> A_neighbors = new HashMap<>();
    private Map<Integer,List<Integer>> B_neighbors = new HashMap<>();


    private List<Integer> A_findNeighbors(int x) {
        List<Integer> nai = A_neighbors.get(x);
        if(nai==null) {
            nai = ChemUtils.findNeighbors(a,x);
            A_neighbors.put(x,nai);
        }
        return nai;
    }
    private List<Integer> B_findNeighbors(int x) {
        List<Integer> nai = B_neighbors.get(x);
        if(nai==null) {
            nai = ChemUtils.findNeighbors(b,x);
            B_neighbors.put(x,nai);
        }
        return nai;
    }

    private BitSet updateForbiddenNodes(int[] m_pre, BitSet f, int vi, int pi) {
        int m[] = new int[m_pre.length];
        for(int zi=0;zi<m.length;zi++){m[zi]=m_pre[zi];}
        m[vi] = pi;

        int minv[] = new int[nvb];
        Arrays.fill(minv,-1);
        for(int zi=0;zi<nvb;zi++) {
            for(int zj=0;zj<nva;zj++) {
                if (m[zj]==zi) {minv[zi]=zj;}
            }
        }

        int matchsize = 0;
        BitSet bsfa = new BitSet(nva);
        BitSet bsfb = new BitSet(nvb);
        boolean bfa[] = new boolean[nva];
        boolean bfb[] = new boolean[nvb];
        //int cntb = 0;
        for(int zi=0;zi<nva;zi++){
            if(m[zi]>=0){
                bfa[zi]=true;
                bfb[m[zi]]=true;
                bsfa.set(zi);
                bsfb.set(m[zi]);
                matchsize++;
            }
            //bfa[zi]=m[zi]>=0;
        }

        if(matchsize>=7) {
            //System.out.println("mkay..");
        }


        StereoMolecule mfa = null;
        StereoMolecule mfb = null;
        int[] amap = null;
        int[] bmap = null;
        int[] a_symmetryranks = null;
        int[] b_symmetryranks = null;

        if(A_subgraphs.containsKey(bsfa)) {
            mfa = A_subgraphs.get(bsfa).getLeft();
            amap = A_subgraphs.get(bsfa).getMiddle();
            a_symmetryranks = A_subgraphs.get(bsfa).getRight();
        }
        else {
            mfa = new StereoMolecule();
            amap = new int[bfa.length];
            a.copyMoleculeByAtoms(mfa,bfa,true,amap);
            Canonizer ca = new Canonizer(mfa,Canonizer.NEGLECT_ANY_STEREO_INFORMATION);
            a_symmetryranks = ca.getSymmetryRanks();
            A_subgraphs.put(bsfa,Triple.of(mfa,a_symmetryranks,amap));
        }

        if(B_subgraphs.containsKey(bsfb)) {
            mfb = B_subgraphs.get(bsfb).getLeft();
            bmap = B_subgraphs.get(bsfb).getMiddle();
            b_symmetryranks = B_subgraphs.get(bsfb).getRight();
        }
        else {
            mfb = new StereoMolecule();
            bmap = new int[bfb.length];
            b.copyMoleculeByAtoms(mfb,bfb,true,bmap);
            Canonizer cb = new Canonizer(mfb,Canonizer.NEGLECT_ANY_STEREO_INFORMATION);
            b_symmetryranks = cb.getSymmetryRanks();
            B_subgraphs.put(bsfb,Triple.of(mfb,b_symmetryranks,bmap));
        }







        if(mfa.getFragments().length>1 || mfb.getFragments().length>1){
            System.out.println("this is very wrong..");
        }


        BitSet fn = (BitSet) f.clone();
        fn.set(vi);
        // for every position in a, try to find available vertex
        // note: we only have to consider the neighbors of v, because these
        //       are the only ones that we can potentially exclude?
        List<Integer> neighbors_v = ChemUtils.findNeighbors(a,vi);
        //List<Integer> neighbors_v = A_findNeighbors(vi);
        for(int zi=0;zi<m.length;zi++) {
            if (fn.get(zi)) {
                continue;
            }
            if (m[zi] >= 0) {
                continue;
            }
            // this one? :)
            if (!neighbors_v.contains(zi)) {
                continue;
            }

            // check if we find u in b such that induced graphs are equal
            boolean found = false;
            List<Integer> neighbors_p = ChemUtils.findNeighbors(b, pi);
            //List<Integer> neighbors_p = B_findNeighbors(pi);

            for (int zb = 0; zb < nvb; zb++) {
                // (probably not..) must be a neighbor of m[zi], otherwise we cannot exclue (?)
                if (!neighbors_p.contains(zb)) {
                    continue;
                }

                //NEARLY:check that bond vi-zi is compatible with pi-zb:
                //ACTUALLY: we have to check that we find a matching from
                //          bonds in a in between zi and already matched atoms in a to
                //          bonds in b in between zb and already matched atoms in b.
                //
                List<Integer> cbonds_a = new ArrayList<>();
                List<Integer> cbonds_b = new ArrayList<>();
                List<Integer> relevant_neighbors_a = ChemUtils.findNeighbors(a, zi).stream().filter(xi -> bfa[xi]).collect(Collectors.toList());
                List<Integer> relevant_neighbors_b = ChemUtils.findNeighbors(b, zb).stream().filter(xi -> bfb[xi]).collect(Collectors.toList());
                //List<Integer> relevant_neighbors_a = A_findNeighbors(zi).stream().filter(xi -> bfa[xi]).collect(Collectors.toList());
                //List<Integer> relevant_neighbors_b = B_findNeighbors(zi).stream().filter(xi -> bfb[xi]).collect(Collectors.toList());

                if (relevant_neighbors_a.size() != relevant_neighbors_b.size()) {
                    // NOT..: now we can exclude:
                    //fn.set(zb);
                    continue;
                }
                // check if we find matching. we just loop over a and check if we find ok vertex in b:
                // (?) it should be ok to do it this way, or is it possible that we miss ok matchings with this implementation??
                boolean matching_ok = true;
                BitSet used_in_b = new BitSet();
                for(int xa=0;xa<cbonds_a.size();xa++) {
                    boolean found_x = false;
                    for(int xb=0;xb<cbonds_b.size();xb++) {
                        if(used_in_b.get(xb)) {continue;}
                        if(a_symmetryranks[ amap[ xa ] ]==b_symmetryranks[ bmap[ xb ] ]) {
                            boolean bond_ok = computeBondCompatibility(a.getBond(vi, zi), b.getBond(pi, zb));
                            if(bond_ok) {
                                found_x = true;
                                used_in_b.set(xb);
                                break;
                            }
                        }
                        else {

                        }
                    }
                    if(!found_x) { matching_ok = false; break; }
                }
                if(!matching_ok) { // i dont believe that the intellij deduction is true???
                    //!!WRONG!! We can only say that this combination did not work..
                    // now we can exclude:
                    //fn.set(zb);
                    continue;
                }
                else {
                    //System.out.println("intellij_is_wrong..");
                    found = true;
                    break;
                }
            }
            if(!found) {
                // now we can exclude:
                fn.set(zi);
            }
        }
        return fn;


            // first, do it simple.
//            if(false) {
//                // check if we find u in b such that induced graphs are equal
//                // i.e. check if vertices are compatible, then if all mapped neighbors are
//                boolean found = false;
//                for (int zb = 0; zb < nvb; zb++) {
//                    //1. check atom compatibility
//                    if (vok.get(zi * zb)) {
//                        continue;
//                    }
//                    //2. check new bonds
//                    boolean bonds_ok = true;
//                    for (int zx = 0; zx < b.getConnAtoms(zb); zx++) {
//
//                        int x_in_b = b.getConnAtom(zb, zx);
//                        int x_in_a = minv[x_in_b];
//
//                    }
//                }
//            }
//        }

    }


    protected boolean computeBondCompatibility(int bonda, int bondb) {
        int sta = a.getBondTypeSimple(bonda);
        int stb = b.getBondTypeSimple(bondb);
        return( sta==stb );
    }

    protected abstract boolean checkComponentConstraintsValid(int m[]);

    protected abstract int selectionStrategy1(BitSet c);

    protected abstract int[] selectionStrategy2(int v, BitSet ci);

    protected abstract int selectionStrategy0(int vmax, BitSet f);

    private BitSet selectCompatibleNodes(int m[], int v, BitSet f) {

        BitSet matched = new BitSet();
        for(int zi=0;zi<m.length;zi++) {if(m[zi]>=0){matched.set(m[zi],true);}}

        BitSet ci = new BitSet(nvb);
        if(f.get(v)) {
            return ci;
        }
        for(int zi=0;zi<nvb;zi++) {
            boolean oki = !matched.get(zi) && vok.get(v*nvb+zi);

            // TODO: do this faster..
            if(oki) {
                boolean test_availability = testCompatibleNodes_simple(m,v,zi);
                if(!test_availability) {
                    //System.out.println("mkay..");
                }
                ci.set(zi,test_availability);
            }
            else {
                ci.set(zi,oki);
            }

        }
        return ci;
    }


    private boolean testCompatibleNodes_simple(int m[], int v, int p) {
        this.stats.countCompatbility();
        //System.out.println("v:"+v+"p:"+p);
        int m2[] = new int[m.length];
        for(int zi=0;zi<m.length;zi++) {
            if(zi==v) {
                if(m[zi]>=0) {
                    //System.out.println("error with adding atom in a to mapping..");
                }
                m2[zi] = p;
            }
            else {
                m2[zi] = m[zi];
            }
        }
        return testMatchingOk_simple(m2);
    }

    private MatchingPerformanceStats stats = new MatchingPerformanceStats();
    public MatchingPerformanceStats getStats() {
        return this.stats;
    }

    public static class MatchingPerformanceStats {
        private int testMatching      = 0;
        private int testCompatibility = 0;
        public void countMatching(){testMatching++;}
        public void countCompatbility(){testCompatibility++;}
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("matching: "+testMatching+"\n");
            sb.append("compatib: "+testCompatibility);
            return sb.toString();
        }
    }

    private Map<Pair<BitSet,BitSet>,Boolean> match_results = new HashMap<>();



    private boolean testMatchingOk_simple(int m[]) {
        stats.countMatching();
        if(false) {
            //StringBuilder sba = new StringBuilder();
            //for(int zi=0;zi<m.length;zi++) { sba.append(zi+ "->" + m[zi] + " ; "); }
            //System.out.println("compare: "+sba.toString());
            BitSet bma = new BitSet(m.length);
            for(int zi=0;zi<m.length;zi++) {bma.set(zi,m[zi]>=0);}
            System.out.println("BS: "+bma.toString());
        }

        int matchsize = 0;
        BitSet bsfa = new BitSet(nva);
        BitSet bsfb = new BitSet(nvb);
        boolean bfa[] = new boolean[nva];
        boolean bfb[] = new boolean[nvb];
        //int cntb = 0;
        for(int zi=0;zi<nva;zi++){
            if(m[zi]>=0){
                bfa[zi]=true;
                bfb[m[zi]]=true;
                bsfa.set(zi);
                bsfb.set(m[zi]);
                matchsize++;
            }
            //bfa[zi]=m[zi]>=0;
        }

        if(matchsize>=7) {
            //System.out.println("mkay..");
        }


        StereoMolecule mfa = null;
        StereoMolecule mfb = null;
        int[] amap = null;
        int[] bmap = null;
        int[] a_symmetryranks = null;
        int[] b_symmetryranks = null;

        if(A_subgraphs.containsKey(bsfa)) {
            mfa = A_subgraphs.get(bsfa).getLeft();
            amap = A_subgraphs.get(bsfa).getMiddle();
            a_symmetryranks = A_subgraphs.get(bsfa).getRight();
        }
        else {
            mfa = new StereoMolecule();
            amap = new int[bfa.length];
            a.copyMoleculeByAtoms(mfa,bfa,true,amap);
            Canonizer ca = new Canonizer(mfa,Canonizer.NEGLECT_ANY_STEREO_INFORMATION);
            a_symmetryranks = ca.getSymmetryRanks();
            A_subgraphs.put(bsfa,Triple.of(mfa,a_symmetryranks,amap));
        }

        if(B_subgraphs.containsKey(bsfb)) {
            mfb = B_subgraphs.get(bsfb).getLeft();
            bmap = B_subgraphs.get(bsfb).getMiddle();
            b_symmetryranks = B_subgraphs.get(bsfb).getRight();
        }
        else {
            mfb = new StereoMolecule();
            bmap = new int[bfb.length];
            b.copyMoleculeByAtoms(mfb,bfb,true,bmap);
            Canonizer cb = new Canonizer(mfb,Canonizer.NEGLECT_ANY_STEREO_INFORMATION);
            b_symmetryranks = cb.getSymmetryRanks();
            B_subgraphs.put(bsfb,Triple.of(mfb,b_symmetryranks,bmap));
        }

        if(match_results.containsKey(Pair.of(bsfa,bsfb))) {
            return match_results.get(Pair.of(bsfa,bsfb));
        }


//        StereoMolecule mfa = new StereoMolecule();
//        StereoMolecule mfb = new StereoMolecule();
//        boolean bfa[] = new boolean[nva];
//        boolean bfb[] = new boolean[nvb];
//        //int cntb = 0;
//        for(int zi=0;zi<nva;zi++){
//            if(m[zi]>=0){
//                bfa[zi]=true;
//                bfb[m[zi]]=true;
//            }
//            //bfa[zi]=m[zi]>=0;
//        }
//        int amap[] = new int[bfa.length];
//        a.copyMoleculeByAtoms(mfa,bfa,true,amap);
//        int bmap[] = new int[bfb.length];
//        b.copyMoleculeByAtoms(mfb,bfb,true,bmap);


        if(false) {
            mfb.setFragment(true);
            //mfa.ensureHelperArrays(Molecule.cHelperCIP);
            //mfb.ensureHelperArrays(Molecule.cHelperCIP);
            if (mfa.getFragments().length > 1 || mfb.getFragments().length > 1) {
                //System.out.println("disjoint_frags..");
                return false;
            }
            SSSearcher sss = new SSSearcher();
            sss.setMol(mfb, mfa);

            boolean isFragIn = sss.isFragmentInMolecule();
            match_results.put(Pair.of(bsfa, bsfb), isFragIn);

            return isFragIn;
        }

        if(true) {
            boolean same = mfa.getIDCode().equals(mfb.getIDCode());
            match_results.put(Pair.of(bsfa, bsfb), same);

            return same;
        }
        //        Canonizer ca = new Canonizer(mfa,Canonizer.NEGLECT_ANY_STEREO_INFORMATION);
//        //int a_symmetryranks[] = ca.getSymmetryRanks();
//
//
//        Canonizer cb = new Canonizer(mfa,Canonizer.NEGLECT_ANY_STEREO_INFORMATION);
//        //int b_symmetryranks[] = cb.getSymmetryRanks();
//        return ca.getIDCode().equals(cb.getIDCode());

        // we should not end up here
        System.out.println("[ERROR] we should not end up here");
        return false;
    }



    private void printDebug2(int[] m, int curmax) {
        StringBuilder sbmap = new StringBuilder();
        for(int zi=0;zi<m.length;zi++) { sbmap.append( zi+":"+m[zi]+" , " ); }
        System.out.println("RESULT: "+curmax+" -> Map: " + sbmap.toString() );
    }
    private boolean printFirstLine = true;
    private void printDebug(int[] m, BitSet c, BitSet f, int vi, int curmax, int upperbound2) {

        StereoMolecule mda = new StereoMolecule(a);
        StereoMolecule mdb = new StereoMolecule(b);
        mda.ensureHelperArrays(Molecule.cHelperCIP);
        mdb.ensureHelperArrays(Molecule.cHelperCIP);
        for(int zi=0;zi<m.length;zi++) {
            if(zi==vi) {
                mda.setAtomCustomLabel(zi,"]V");
            }
            if(c.get(zi)) {
                mda.setAtomCustomLabel(zi,"]C");
            }
            if(f.get(zi)) {
                mda.setAtomCustomLabel(zi,"]F");
            }
            if(m[zi]>=0) {
                mda.setAtomColor(zi,Molecule.cAtomColorMagenta);
                mda.setAtomCustomLabel(zi,"¨]"+zi);
                mdb.setAtomColor(m[zi],Molecule.cAtomColorMagenta);
                mdb.setAtomCustomLabel(m[zi],"]"+zi);
            }
        }
        List<String> parts = new ArrayList<>();
        parts.add(new Canonizer(mda,Canonizer.ENCODE_ATOM_CUSTOM_LABELS).getIDCode());
        parts.add(new Canonizer(mdb,Canonizer.ENCODE_ATOM_CUSTOM_LABELS).getIDCode());
        parts.add(""+curmax);
        parts.add(""+upperbound2);

        if(printFirstLine) {
            System.out.print("\n"+"ma[idcode]\tmb[idcode]\tcurmax\tub2");
            printFirstLine=false;
        }
        System.out.print("\n"+String.join(",\t",parts));
    }

    public static List<StereoMolecule> getSomeTestMolecules(int n) {

        //StereoMolecule ma = ChemUtils.parseIDCode("fhy`B@N@dDfYYwyVBdkpX@@BjhH@@");
        //StereoMolecule mb = ChemUtils.parseIDCode("gFp@DiTt@@@");
        //StereoMolecule mb = ChemUtils.parseIDCode("dif@`FBHRYVZZ@B`@@");

        //StereoMolecule ma = ChemUtils.parseIDCode("fhy@B@@RFQQQJKHjILyKW`p@@QUPP@@");
        //StereoMolecule mb = ChemUtils.parseIDCode("fag@B@@QFQQQJKIJZIJHqgIZbdmV}`@@bjjjhJ`@@");
        //StereoMolecule mc = ChemUtils.parseIDCode("dg|@P@dKdLbbRabbMLip@UUQ@@");

        List<StereoMolecule> mols = new ArrayList<>();
        try(BufferedReader in_a = new BufferedReader(new FileReader("c:\\datasets\\idcodes\\rand_20k_from_cm_avail.txt"))) {
            String line_a = null;
            while( (line_a=in_a.readLine()) != null && mols.size()<n) {
                mols.add(ChemUtils.parseIDCode(line_a));
            }
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        //mols.add(ma);
        //mols.add(mb);
        //mols.add(mc);
        return mols;
    }

}
