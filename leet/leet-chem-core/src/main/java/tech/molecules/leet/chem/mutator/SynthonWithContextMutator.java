package tech.molecules.leet.chem.mutator;

import com.actelion.research.chem.StereoMolecule;

import java.util.List;

public class SynthonWithContextMutator implements Mutator<SynthonWithContext,SynthonWithContext,int[][]> {

    @Override
    public List<int[][]> findPossibleMutationConfigurations(SynthonWithContext seed, SynthonWithContext mutation) {
        return seed.computePossibleAssemblies(mutation);
    }

    @Override
    public StereoMolecule applyMutation(SynthonWithContext seed, SynthonWithContext mutation, int[][] config) {
        return SynthonWithContext.annealSynthons(seed,mutation,config);
    }
}
