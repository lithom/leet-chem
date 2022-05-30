package tech.molecules.leet.chem.mutator;

import com.actelion.research.chem.StereoMolecule;

import java.util.List;

public class StereoMoleculeMutator implements Mutator<StereoMolecule,SynthonWithContext,int[][]> {


    @Override
    public List<int[][]> findPossibleMutationConfigurations(StereoMolecule seed, SynthonWithContext mutation) {
        throw new Error("not yet implemented");
        //return null;
    }

    @Override
    public StereoMolecule applyMutation(StereoMolecule seed, SynthonWithContext mutation, int[][] config) {
        return null;
    }
}
