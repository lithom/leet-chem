package tech.molecules.deep;

import com.actelion.research.chem.StereoMolecule;

import java.util.BitSet;

public class PathEncoder {

    private StereoMolecule m;
    private int a,b;

    public PathEncoder(StereoMolecule m, int a, int b) {
        this.m = m;
        this.a = a;
        this.b = b;
    }

    public BitSet encode() {
        return null;
    }

}
