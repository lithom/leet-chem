package tech.molecules.leet.chem.mutator;

import com.actelion.research.chem.StereoMolecule;

import java.util.List;

public interface Mutator<S,M,C> {

    public List<C> findPossibleMutationConfigurations(S seed, M mutation);
    public StereoMolecule applyMutation(S seed, M mutation, C config);

}
