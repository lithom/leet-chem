package tech.molecules.leet.chem.mutator;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.chemicalspaces.synthon.SynthonReactor;

import java.util.ArrayList;
import java.util.List;

public interface SynthonWithContext {

    public StereoMolecule getSynthon();
    public StereoMolecule getContext();
    public int[][] getMapFromSynthonConnectorsToContextConnectors();

    /**
     *
     * @param other
     * @return map from this synthon molecule connectors to other synthon molecule connectors.
     */
    public List<int[][]> computePossibleAssemblies(SynthonWithContext other);


    /**
     *
     * @param a
     * @param b
     * @param assembly
     * @return
     */
    public static StereoMolecule annealSynthons(SynthonWithContext a, SynthonWithContext b, int[][] assembly) {
        //boolean is_fragment = a.getSynthon().isFragment() || b.getSynthon().isFragment();
        //StereoMolecule ma = new StereoMolecule();
        //ma.setFragment(is_fragment);

        //int pa[] = new int[a.getSynthon().getAtoms()];
        //int pb[] = new int[b.getSynthon().getAtoms()];
        //ma.addFragment(a.getSynthon(),0,pa);
        //ma.addFragment(b.getSynthon(),0,pb);

        StereoMolecule a1 = new StereoMolecule(a.getSynthon());
        a1.ensureHelperArrays(Molecule.cHelperCIP);
        StereoMolecule a2 = new StereoMolecule(b.getSynthon());
        a2.ensureHelperArrays(Molecule.cHelperCIP);
        for(int zi=0;zi<assembly.length;zi++) {
            a1.setAtomicNo(assembly[zi][0],92+zi);
            a2.setAtomicNo(assembly[zi][1],92+zi);
        }

        List<StereoMolecule> frags = new ArrayList<>();
        frags.add(a1);
        frags.add(a2);

        StereoMolecule assembled = SynthonReactor.react(frags);
        assembled.ensureHelperArrays(Molecule.cHelperCIP);
        return assembled;
    }

}
