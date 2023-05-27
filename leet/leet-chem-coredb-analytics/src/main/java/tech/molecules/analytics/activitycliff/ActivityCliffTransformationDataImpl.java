package tech.molecules.analytics.activitycliff;

import tech.molecules.analytics.MMPInstance;
import tech.molecules.analytics.MMPTransformation;
import tech.molecules.analytics.NumericalMMPInstance;

import java.util.List;

public class ActivityCliffTransformationDataImpl implements ActivityCliffTransformationData {

    private MMPTransformation transformation;
    private List<NumericalMMPInstance> mmps;
    private ActivityCliffDefinition activityCliffDefinition;

    private ActivityCliffProbabilityCalculator.ACData activityCliffData;

    public ActivityCliffTransformationDataImpl(MMPTransformation transformation, List<NumericalMMPInstance> mmps, ActivityCliffDefinition activityCliffDefinition, ActivityCliffProbabilityCalculator.ACData activityCliffData) {
        this.transformation = transformation;
        this.mmps = mmps;
        this.activityCliffDefinition = activityCliffDefinition;
        this.activityCliffData = activityCliffData;
    }

    @Override
    public MMPTransformation getTransformation() {
        return transformation;
    }

    @Override
    public List<NumericalMMPInstance> getMMPInstances() {
        return mmps;
    }

    @Override
    public ActivityCliffDefinition getActivityCliffDefinition() {
        return activityCliffDefinition;
    }

    @Override
    public ActivityCliffProbabilityCalculator.ACData getActivityCliffData() {
        return activityCliffData;
    }

}
