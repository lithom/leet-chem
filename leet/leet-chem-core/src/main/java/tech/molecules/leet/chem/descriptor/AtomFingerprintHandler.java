package tech.molecules.leet.chem.descriptor;

import com.actelion.research.chem.StereoMolecule;

public interface AtomFingerprintHandler<T> {

    public T[] createDescriptor(StereoMolecule mi);
    public double computeSimilarity(T a,T b);

}
