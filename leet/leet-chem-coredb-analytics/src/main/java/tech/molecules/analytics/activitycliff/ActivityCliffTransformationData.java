package tech.molecules.analytics.activitycliff;

import tech.molecules.analytics.MMPTransformation;
import tech.molecules.analytics.NumericalMMPInstance;

import java.util.List;

public interface ActivityCliffTransformationData {


    public MMPTransformation getTransformation();
    public List<NumericalMMPInstance> getMMPInstances();
    public ActivityCliffDefinition getActivityCliffDefinition();

    public ACData getActivityCliffData();

    class ACData {
        public final double probability;
        public final int nTransformations;
        public final int nAHiBLo;
        public final int nALoBHi;

        public ACData(double probability, int nTransformations, int nAHiBLo, int nALoBHi) {
            this.probability = probability;
            this.nTransformations = nTransformations;
            this.nAHiBLo = nAHiBLo;
            this.nALoBHi = nALoBHi;
        }
    }
}
