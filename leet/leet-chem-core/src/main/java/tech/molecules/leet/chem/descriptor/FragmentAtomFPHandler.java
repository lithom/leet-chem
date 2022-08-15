package tech.molecules.leet.chem.descriptor;

import com.actelion.research.chem.Canonizer;
import com.actelion.research.chem.CanonizerUtil;
import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.mcs.MCSFast;
import com.actelion.research.chem.shredder.Fragment;
import com.actelion.research.chem.shredder.FragmentGenerator;
import tech.molecules.leet.chem.BitSetUtils;

import java.util.*;

public class FragmentAtomFPHandler implements AtomFingerprintHandler<BitSet> {

    private int minsize;
    private int maxsize;

    private int hashBits;

    public FragmentAtomFPHandler(int minsize, int maxsize, int hashbits) {
        this.minsize = minsize;
        this.maxsize = maxsize;
        this.hashBits = hashbits;
    }

    @Override
    public BitSet[] createDescriptor(StereoMolecule mi) {
        mi.ensureHelperArrays(Molecule.cHelperCIP);
        BitSet[] fps = new BitSet[mi.getAtoms()];
        for(int zi=0;zi<mi.getAtoms();zi++) { fps[zi] = new BitSet(this.hashBits); }
        FragmentGenerator ff = new FragmentGenerator(mi,this.minsize,this.maxsize);
        ff.computeFragments();
        StereoMolecule fi = new StereoMolecule();
        List<BitSet> frags = ff.getFragments();
        List<boolean[]> frags_boolean = ff.getFragmentsAsBooleanArrays();
        for(int zi=0;zi<frags.size();zi++) {
            mi.copyMoleculeByAtoms(fi,frags_boolean.get(zi),true,null);
            int hi = (int) Math.abs( (CanonizerUtil.getNoStereoHash(fi,false) % hashBits ) );
            for(int ai : frags.get(zi).stream().toArray() )  {
                fps[ai].set(hi,true);
            }
        }
        //Canonizer ca = CanonizerUtil.getNoStereoHash();
        MCSFast mcsf = new MCSFast();

        return fps;
    }

    @Override
    public double computeSimilarity(BitSet a, BitSet b) {

        return BitSetUtils.tanimoto_similarity(a,b);
    }
}
