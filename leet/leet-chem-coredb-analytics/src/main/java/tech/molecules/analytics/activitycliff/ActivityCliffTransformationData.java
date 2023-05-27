package tech.molecules.analytics.activitycliff;

import tech.molecules.analytics.MMPInstance;
import tech.molecules.analytics.MMPTransformation;
import tech.molecules.analytics.NumericalMMPInstance;

import java.util.List;

public interface ActivityCliffTransformationData {


    public MMPTransformation getTransformation();
    public List<NumericalMMPInstance> getMMPInstances();
    public ActivityCliffDefinition getActivityCliffDefinition();

    public ActivityCliffProbabilityCalculator.ACData getActivityCliffData();

}
