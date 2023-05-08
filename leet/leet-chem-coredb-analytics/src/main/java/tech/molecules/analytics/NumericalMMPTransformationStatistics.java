package tech.molecules.analytics;

import java.util.List;

public interface NumericalMMPTransformationStatistics {

    public MMPTransformation getTransformation();

    public String getAttributeName();

    public int getAssayID();

    public List<NumericalMMPInstance> getInstances();

    default public int getN() {
        return getInstances().size();
    }

    default public int getN_Up() {
        return (int) getInstances().stream().filter(xi -> xi.getShiftAtoB() > 0).count();
    }

    default public int getN_Down() {
        return (int) getInstances().stream().filter(xi -> xi.getShiftAtoB() < 0).count();
    }

}
