package tech.molecules.analytics;

import java.util.List;

public class NumericalMMPTransformationStatisticsImpl implements NumericalMMPTransformationStatistics {

    private MMPTransformation transformation;

    private int assayID;
    private String attribute;

    private List<NumericalMMPInstance> instances;

    public NumericalMMPTransformationStatisticsImpl(MMPTransformation transformation, int assayID, String attribute, List<NumericalMMPInstance> instances) {
        this.transformation = transformation;
        this.assayID = assayID;
        this.attribute = attribute;
        this.instances = instances;
    }

    @Override
    public MMPTransformation getTransformation() {
        return null;
    }

    @Override
    public String getAttributeName() {
        return attribute;
    }

    @Override
    public int getAssayID() {
        return assayID;
    }

    @Override
    public List<NumericalMMPInstance> getInstances() {
        return instances;
    }
}
