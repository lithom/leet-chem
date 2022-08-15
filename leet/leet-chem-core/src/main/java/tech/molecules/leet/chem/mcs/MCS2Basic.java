package tech.molecules.leet.chem.mcs;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.mcs.MCSFast;
import tech.molecules.leet.chem.BitSetUtils;
import tech.molecules.leet.chem.descriptor.FragmentAtomFPHandler;

import java.util.BitSet;
import java.util.List;
import java.util.stream.Collectors;

public class MCS2Basic extends MCS2 {
    @Override
    protected boolean checkComponentConstraintsValid(int[] m) {
        return true;
    }


    @Override
    protected int selectionStrategy0(int max,BitSet f) {
        double vmax = -1;
        int    imax = -1;
        for(int vi =0;vi<max;vi++) {
            if(f.get(vi)){continue;}
            if(bestMatchValueInA[vi]>vmax) {
                vmax = bestMatchValueInA[vi];
                imax = vi;
            }
        }
        return imax;
    }

    @Override
    protected int selectionStrategy1(BitSet c) {
        //return c.stream().findFirst().getAsInt();
        //List<Integer> list = c.stream().boxed().collect(Collectors.toList());
        //list.sort( (xi,yi) -> -Double.compare( bestMatchValueInA[xi] , bestMatchValueInA[yi] ) );
        //int[] result =  list.stream().mapToInt( li -> li ).toArray();
        double vmax = -1;
        int    imax = -1;
        for(int vi : c.stream().toArray()) {
            if(bestMatchValueInA[vi]>vmax) {
                vmax = bestMatchValueInA[vi];
                imax = vi;
            }
        }
        return imax;
    }

    @Override
    protected int[] selectionStrategy2(int v, BitSet ci) {
        List<Integer> list = ci.stream().boxed().collect(Collectors.toList());
        list.sort( (xi,yi) -> -Double.compare( similarities[v][xi] , similarities[v][yi] ) );
        int[] result =  list.stream().mapToInt( li -> li ).toArray();
        return result;
    }


    private BitSet[] descriptor_a = null;
    private BitSet[] descriptor_b = null;

    /**
     * atom similarities
     */
    private double[][] similarities = null;

    /**
     * Position i contains highest match value for atom i in molecule a.
     */
    private double[] bestMatchValueInA = null;

    public void setAtomDescriptors(BitSet[] a, BitSet[] b) {
        this.descriptor_a = a;
        this.descriptor_b = b;

        this.similarities = new double[a.length][b.length];
        this.bestMatchValueInA = new double[a.length];
        // compute similarity stuff:
        for(int zi=0;zi<a.length;zi++) {
            double bestmatch_a = -1;
            for(int zj=0;zj<b.length;zj++) {
                double simij = BitSetUtils.tanimoto_similarity(a[zi],b[zj]);
                similarities[zi][zj] = simij;
                bestmatch_a = Math.max(bestmatch_a,simij);
            }
            bestMatchValueInA[zi] = bestmatch_a;
        }
    }

    public static void main(String args[]) {
        List<StereoMolecule> mols = MCS2.getSomeTestMolecules(120).stream().filter(mi -> mi.getAtoms()<40).collect(Collectors.toList());

        FragmentAtomFPHandler afph = new FragmentAtomFPHandler(3,4,512);

        for(int zi=0;zi<100;zi++) {
            long ta = System.currentTimeMillis();
            MCS2Basic mcs = new MCS2Basic();
            StereoMolecule ma = mols.get(zi);
            StereoMolecule mb = mols.get(zi+1);
            mcs.setAB(ma, mb);
            mcs.setAtomDescriptors(afph.createDescriptor(ma),afph.createDescriptor(mb));
            mcs.computeMCS();
            long tb = System.currentTimeMillis();

            long t2a = System.currentTimeMillis();
            MCSFast mcsa = new MCSFast();
            mcsa.set(ma, mb);
            StereoMolecule mresult = mcsa.getMCS();
            mresult.ensureHelperArrays(Molecule.cHelperCIP);
            long t2b = System.currentTimeMillis();
            System.out.println("max matching B: "+mresult.getAtoms());
            System.out.println("t1= " + (tb - ta) + "   t2= " + (t2b - t2a));

            System.out.println("\nPerformance stats:\n"+mcs.getStats().toString());
            System.out.println("\n");
        }


    }
}
