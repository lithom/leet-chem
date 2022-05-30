package tech.molecules.leet.chem.mutator;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.shredder.FragmentDecomposition;

import java.util.ArrayList;
import java.util.List;

public class FragmentDecompositionSynthon implements SynthonWithContext {

    private FragmentDecomposition decomp;
    private StereoMolecule synthon;
    private StereoMolecule context;

    private int[][] mapSynthonConnectorsToContextConnectors;


    public FragmentDecompositionSynthon(FragmentDecomposition decomp) {
        //this.synthon = new StereoMolecule( decomp.getCentralFrag() );
        int cpos[][] = new int[decomp.getNumberOfConnectors()][2];
        StereoMolecule combined = decomp.createCombinedFragmentsMoleculeWithLinkerConnectors(cpos,false);
        this.synthon = new StereoMolecule();
        this.synthon.setFragment(combined.isFragment());

        // central: only one add fragment needed, for the context we have to separately copy all fragments
        int map_central[] = new int[combined.getAtoms()];
        this.synthon.addFragment(combined,cpos[0][0],map_central);
        this.synthon.ensureHelperArrays(Molecule.cHelperCIP);

        this.context = new StereoMolecule();
        this.context.setFragment(combined.isFragment());
        List<int[]> maps_context = new ArrayList<>();
        for(int zi=0;zi<cpos.length;zi++) {
            int map_cont_i[] = new int[combined.getAtoms()];
            this.context.addFragment(combined,cpos[zi][1],map_cont_i);
            maps_context.add(map_cont_i);
        }
        this.context.ensureHelperArrays(Molecule.cHelperCIP);

        this.mapSynthonConnectorsToContextConnectors = new int[cpos.length][2];
        for(int ci = 0;ci<cpos.length;ci++) {
            int pos_a = map_central[cpos[ci][0]];
            int pos_b = maps_context.get(ci)[cpos[ci][1]];
            this.mapSynthonConnectorsToContextConnectors[ci][0] = pos_a;
            this.mapSynthonConnectorsToContextConnectors[ci][1] = pos_b;
        }
        this.decomp = decomp;
    }

    @Override
    public StereoMolecule getSynthon() {
        return this.synthon;
    }

    @Override
    public StereoMolecule getContext() {
        return this.context;
    }

    @Override
    public int[][] getMapFromSynthonConnectorsToContextConnectors() {
        return this.mapSynthonConnectorsToContextConnectors;
    }

    @Override
    public List<int[][]> computePossibleAssemblies(SynthonWithContext other) {
        return null;
    }

}
